import argparse
import atexit
import copy
import gc
import itertools
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.entrypoint_utils import load_yaml_config


LOCAL_EVAL_SEQUENCE_LENGTH = 2048
C4_STREAMING_SAMPLE_COUNT = 1100
C4_MAX_TOKENS = 256 * LOCAL_EVAL_SEQUENCE_LENGTH


# Local HuggingFace path. Used for the PPL check that runs before evalscope.


@torch.no_grad()
def evaluate_ppl(model: nn.Module,
                 tokenizer: AutoTokenizer,
                 eval_ppl: str) -> Dict[str, float]:
    """Compute the perplexity of an in-memory model on comma-separated Hugging Face datasets.

    Each entry may carry an ``@split`` suffix. C4 entries stream the first
    ``C4_STREAMING_SAMPLE_COUNT`` rows of the ``en`` config (default split
    ``validation``); other datasets use their ``test`` split by default. The
    text is concatenated and scored in non-overlapping windows of
    ``LOCAL_EVAL_SEQUENCE_LENGTH`` tokens.

    Args:
        model: Causal LM to score.
        tokenizer: Tokenizer matching ``model``.
        eval_ppl: Comma-separated dataset entries.

    Returns:
        Mapping from dataset entry to perplexity.
    """
    seqlen = LOCAL_EVAL_SEQUENCE_LENGTH
    device = model.device
    ppl_results = {}
    for dataset_spec in eval_ppl.split(","):
        # Allow "dataset@split" for datasets whose default split is not valid for PPL.
        if "@" in dataset_spec.split("/")[-1]:
            dataset, split_override = dataset_spec.rsplit("@", 1)
        else:
            dataset, split_override = dataset_spec, None

        result_key = dataset_spec

        if "c4" in dataset:
            raw_dataset = load_dataset(
                dataset,
                "en",
                split=split_override if split_override else "validation",
                streaming=True,
            )
            sampled_rows = list(itertools.islice(raw_dataset, C4_STREAMING_SAMPLE_COUNT))
            tokenized = tokenizer(" ".join([row["text"] for row in sampled_rows]), return_tensors="pt")
            token_ids = tokenized.input_ids[:, :C4_MAX_TOKENS]
        else:
            raw_dataset = load_dataset(
                dataset,
                split=split_override if split_override else "test",
            )
            tokenized = tokenizer("\n\n".join(raw_dataset["text"]), return_tensors="pt")
            token_ids = tokenized.input_ids

        sample_count = token_ids.numel() // seqlen
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()
        nlls = []
        for sample_idx in tqdm(range(sample_count)):
            start = sample_idx * seqlen
            end = (sample_idx + 1) * seqlen
            batch = token_ids[:, start:end].to(device)
            outputs = model(input_ids=batch)
            shift_logits = outputs.logits[..., :-1, :].float().contiguous()
            shift_labels = batch[..., 1:].contiguous().to(shift_logits.device)
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss * seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (sample_count * seqlen))
        print(f"[PPL] {result_key}: {ppl.item()}")
        model.config.use_cache = use_cache
        ppl_results[result_key] = ppl.item()

    return ppl_results


# vLLM/evalscope path. Used by script/eval_thinking_vllm*.py.


ALL_DATASET_CONFIGS = {
    "arc": {"args": {"few_shot_num": 0}, "gen": "mcq"},
    "piqa": {"args": {"few_shot_num": 0}, "gen": "mcq"},
    "winogrande": {"args": {"few_shot_num": 0}, "gen": "mcq"},
    "gsm8k": {"args": {"few_shot_num": 4}, "gen": "mcq"},
    "humaneval": {"args": {"few_shot_num": 0, "aggregation": "mean_and_pass_at_k"}, "gen": "mcq"},
    "live_code_bench": {"args": {"few_shot_num": 0}, "gen": "mcq"},
    "math_500": {"args": {"few_shot_num": 0, "aggregation": "mean_and_pass_at_k"}, "gen": "math"},
    "aime25": {"args": {"few_shot_num": 0, "aggregation": "mean_and_pass_at_k"}, "gen": "aime"},
    "aime26": {"args": {"few_shot_num": 0, "aggregation": "mean_and_pass_at_k"}, "gen": "aime"},
}

DEFAULT_PPL_DATASETS = (
    "Self-GRIT/wikitext-2-raw-v1-preprocessed@test,"
    "allenai/c4"
)

_CLUSTER_KEY_PATTERN = re.compile(
    r"\.(centroids|A_assignment|B_assignment|assignment_index)$"
)


@dataclass
class VllmEvalConfig:
    """Settings for one vLLM + evalscope evaluation run."""

    model_dir: Optional[str]
    model_type: str
    act_quant_bits: int
    task_list: List[str]
    math_gen_config: Dict[str, Any]
    aime_gen_config: Dict[str, Any]
    mcq_gen_config: Dict[str, Any]
    vllm_port: int
    clean_tmp_default: Optional[str]
    tokenizer_class_policy: str
    tokenizer_path: Optional[str] = None
    limit: int = -1
    seed: int = 42
    eval_batch_size: int = 32
    vllm_gpu_mem_util: float = 0.90
    vllm_max_model_len: int = 32768
    vllm_tensor_parallel: int = 2
    vllm_max_num_seqs: int = 64
    ppl_eval_datasets: str = DEFAULT_PPL_DATASETS
    dataset_configs: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: copy.deepcopy(ALL_DATASET_CONFIGS)
    )

    def generation_configs(self) -> Dict[str, Dict[str, Any]]:
        return {
            "mcq": copy.deepcopy(self.mcq_gen_config),
            "math": copy.deepcopy(self.math_gen_config),
            "aime": copy.deepcopy(self.aime_gen_config),
        }

    def repeats_per_task(self) -> Dict[str, int]:
        return {task: (8 if task.startswith("aime") else 1) for task in self.task_list}


def default_cache_root() -> Path:
    root = os.environ.get("CLUSQUANT_CACHE_ROOT") or os.environ.get("VLLM_CACHE_ROOT")
    if root:
        return Path(root)

    user = os.environ.get("USER")
    scratch_root = Path("/scratch") / user / ".cache" if user else None
    if scratch_root and scratch_root.parent.exists():
        return scratch_root

    return Path.home() / ".cache"


def default_cache_env() -> Dict[str, str]:
    cache_root = default_cache_root()
    hf_home = cache_root / "huggingface"
    return {
        "HF_HOME": str(hf_home),
        "TRANSFORMERS_CACHE": str(hf_home / "hub"),
        "MODELSCOPE_CACHE": str(cache_root / "modelscope"),
        "VLLM_CACHE_ROOT": str(cache_root),
    }


def set_cache_env() -> None:
    override_from_root = bool(os.environ.get("CLUSQUANT_CACHE_ROOT"))
    for key, value in default_cache_env().items():
        if override_from_root:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


def strip_cluster_weights(src_dir: str, config: VllmEvalConfig) -> str:
    """Copy a model dir and remove ClusterLinear-only weights that vLLM cannot load."""
    import safetensors.torch as st

    source_dir = Path(src_dir)
    tmp_root = os.environ.get("VLLM_CLEAN_TMPDIR") or config.clean_tmp_default
    if not tmp_root:
        raise ValueError("clean tmp directory is required")
    os.makedirs(tmp_root, exist_ok=True)
    clean_model_dir = Path(tempfile.mkdtemp(prefix="vllm_clean_", dir=tmp_root))
    atexit.register(shutil.rmtree, clean_model_dir, ignore_errors=True)

    for path in source_dir.iterdir():
        if path.suffix != ".safetensors":
            (clean_model_dir / path.name).symlink_to(path.resolve())

    removed_keys = set()
    for shard in sorted(source_dir.glob("*.safetensors")):
        weights = st.load_file(str(shard))
        clean = {key: value for key, value in weights.items() if not _CLUSTER_KEY_PATTERN.search(key)}
        removed_keys.update(key for key in weights if _CLUSTER_KEY_PATTERN.search(key))
        st.save_file(clean, str(clean_model_dir / shard.name))

    source_index_path = source_dir / "model.safetensors.index.json"
    clean_index_path = clean_model_dir / "model.safetensors.index.json"
    if clean_index_path.is_symlink():
        clean_index_path.unlink()
    if source_index_path.exists():
        with open(source_index_path) as f:
            index = json.load(f)
        index["weight_map"] = {
            key: value
            for key, value in index["weight_map"].items()
            if not _CLUSTER_KEY_PATTERN.search(key)
        }
        with open(clean_index_path, "w") as f:
            json.dump(index, f)

    if removed_keys:
        print(f"[INFO] Stripped {len(removed_keys)} ClusterLinear key(s)")

    _fix_tokenizer_config(clean_model_dir, config.tokenizer_class_policy)
    return str(clean_model_dir)


def _fix_tokenizer_config(model_dir: Path, tokenizer_class_policy: str) -> None:
    tok_cfg_path = model_dir / "tokenizer_config.json"
    if tok_cfg_path.is_symlink():
        real = tok_cfg_path.resolve()
        tok_cfg_path.unlink()
        shutil.copy2(real, tok_cfg_path)
    if not tok_cfg_path.exists():
        return

    with open(tok_cfg_path) as f:
        tok_cfg = json.load(f)

    changed = False
    if isinstance(tok_cfg.get("extra_special_tokens"), list):
        tok_cfg["extra_special_tokens"] = {}
        changed = True
        print("[INFO] Fixed extra_special_tokens in tokenizer_config.json (list -> dict)")

    if tok_cfg.get("tokenizer_class") == "TokenizersBackend":
        if tokenizer_class_policy == "replace":
            tok_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            print("[INFO] Replaced tokenizer_class 'TokenizersBackend' with 'PreTrainedTokenizerFast'")
        elif tokenizer_class_policy == "remove":
            tok_cfg.pop("tokenizer_class")
            print("[INFO] Removed invalid tokenizer_class 'TokenizersBackend' from tokenizer_config.json")
        elif tokenizer_class_policy != "keep":
            raise ValueError(f"Unknown tokenizer_class_policy={tokenizer_class_policy!r}")
        changed = tokenizer_class_policy != "keep"

    if changed:
        with open(tok_cfg_path, "w") as f:
            json.dump(tok_cfg, f, ensure_ascii=False, indent=2)


def _setup_sitecustomize(config: VllmEvalConfig) -> str:
    """Create a temporary sitecustomize.py that registers local vLLM plugins."""
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(utils_dir)
    plugin_dir = os.path.join(project_root, "script")

    site_code = f'''
import os, sys, traceback

for p in [{plugin_dir!r}, {utils_dir!r}, {project_root!r}]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["ACT_QUANT_BITS"] = {str(config.act_quant_bits)!r}

try:
    import act_quant_plugin
except Exception as exc:
    print("[WARN] failed to import act_quant_plugin:", exc, file=sys.stderr)
    traceback.print_exc()

try:
    import transformers.tokenization_utils_base as _tub
    _orig = _tub.PreTrainedTokenizerBase._set_model_specific_special_tokens
    def _patched(self, special_tokens):
        if isinstance(special_tokens, list):
            special_tokens = {{}}
        return _orig(self, special_tokens)
    _tub.PreTrainedTokenizerBase._set_model_specific_special_tokens = _patched
except Exception:
    pass
'''
    tmp_site_dir = os.path.join(tempfile.gettempdir(), "vllm_site")
    os.makedirs(tmp_site_dir, exist_ok=True)
    with open(os.path.join(tmp_site_dir, "sitecustomize.py"), "w") as f:
        f.write(site_code)
    atexit.register(shutil.rmtree, tmp_site_dir, ignore_errors=True)
    return tmp_site_dir


def start_vllm_server(
    model_dir: str,
    quantization: Optional[str],
    config: VllmEvalConfig,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_dir,
    ]
    if config.tokenizer_path:
        cmd += ["--tokenizer", config.tokenizer_path]
    cmd += [
        "--served-model-name",
        config.model_type,
        "--port",
        str(config.vllm_port),
        "--max-model-len",
        str(config.vllm_max_model_len),
        "--gpu-memory-utilization",
        str(config.vllm_gpu_mem_util),
        "--tensor-parallel-size",
        str(config.vllm_tensor_parallel),
        "--max-num-seqs",
        str(config.vllm_max_num_seqs),
        "--dtype",
        "bfloat16",
        "--trust-remote-code",
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--reasoning-parser",
        "deepseek_r1",
    ]
    if quantization:
        cmd += ["--quantization", quantization]

    env = os.environ.copy()
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(utils_dir)
    plugin_dir = os.path.join(project_root, "script")
    env["PYTHONPATH"] = plugin_dir + ":" + utils_dir + ":" + project_root + ":" + env.get("PYTHONPATH", "")

    if quantization:
        site_dir = _setup_sitecustomize(config)
        env["PYTHONPATH"] = site_dir + ":" + env["PYTHONPATH"]

    print(f"[INFO] Starting vLLM server on port {config.vllm_port}")
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)

    def cleanup():
        if proc.poll() is None:
            print("[INFO] Shutting down vLLM server...")
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(cleanup)
    return proc


def wait_for_server(port: int, proc: subprocess.Popen, timeout: int = 1200) -> None:
    import urllib.request

    url = f"http://127.0.0.1:{port}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited early with code {proc.returncode}; "
                f"check stderr for the real error"
            )
        try:
            urllib.request.urlopen(url, timeout=2)
            print(f"[INFO] vLLM server ready on port {port} (took {time.time() - start:.1f}s)")
            return
        except Exception:
            time.sleep(3)
    raise RuntimeError(f"vLLM server did not start within {timeout}s")


def run_pre_eval_ppl(model_dir: str, config: VllmEvalConfig) -> None:
    """Run local PPL before evalscope so model quality regressions are visible early."""
    if not config.ppl_eval_datasets:
        return

    print(f"[INFO] Running pre-eval PPL on {config.ppl_eval_datasets}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer_source = config.tokenizer_path or model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    evaluate_ppl(model, tokenizer, eval_ppl=config.ppl_eval_datasets)
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_eval_cli(config: VllmEvalConfig) -> None:
    """Parse the evaluation command line, start vLLM for ``config`` and run evalscope."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None,
        help="experiment YAML; use bits.activation for downstream activation quantization",
    )
    parser.add_argument("--model-dir", type=str, default=None, help="path to the model directory")
    parser.add_argument(
        "--clean-tmp-dir",
        type=str,
        default=None,
        help="directory for temporary vLLM-clean model copies",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help="root for HF/modelscope/vLLM caches; overrides derived cache env vars",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="tokenizer directory passed to vLLM instead of the model directory",
    )
    parser.add_argument("--no-quantization", action="store_true", help="disable activation quantization")
    parser.add_argument("--skip-ppl", action="store_true", help="skip pre-eval PPL evaluation")
    parser.add_argument("--server-only", action="store_true", help="only start vLLM server, do not run evaluation")
    args = parser.parse_args()

    if args.config:
        experiment_config = load_yaml_config(args.config)
        activation_bits = int(experiment_config["bits"]["activation"])
        if activation_bits not in (4, 8):
            parser.error("config bits.activation must be 4 or 8; use --no-quantization to disable")
        config = replace(config, act_quant_bits=activation_bits)

    if args.cache_root:
        os.environ["CLUSQUANT_CACHE_ROOT"] = args.cache_root
    set_cache_env()

    model_dir = args.model_dir or config.model_dir
    clean_tmp_dir = args.clean_tmp_dir or os.environ.get("VLLM_CLEAN_TMPDIR") or config.clean_tmp_default
    if not model_dir:
        parser.error("--model-dir is required")
    if not clean_tmp_dir:
        parser.error("--clean-tmp-dir is required")

    config = replace(
        config,
        model_dir=model_dir,
        clean_tmp_default=clean_tmp_dir,
        tokenizer_path=args.tokenizer_path or config.tokenizer_path,
    )
    if args.clean_tmp_dir:
        os.environ["VLLM_CLEAN_TMPDIR"] = args.clean_tmp_dir

    quantization = None if (args.no_quantization or config.act_quant_bits == -1) else "act_quant"
    stripped_model_dir = strip_cluster_weights(model_dir, config)

    print(f"[INFO] Evaluating model={model_dir}, quantization={quantization or 'none'}")
    print(f"[INFO] VLLM_CACHE_ROOT={os.environ.get('VLLM_CACHE_ROOT')}")
    if not args.skip_ppl and not args.server_only:
        run_pre_eval_ppl(stripped_model_dir, config)

    server_proc = start_vllm_server(stripped_model_dir, quantization, config)
    wait_for_server(config.vllm_port, server_proc)

    if args.server_only:
        print(f"[INFO] Server running on port {config.vllm_port}. Press Ctrl+C to stop.")
        try:
            server_proc.wait()
        except KeyboardInterrupt:
            pass
        sys.exit(0)

    run_evalscope(config)


def run_evalscope(config: VllmEvalConfig) -> None:
    """Run every task in ``config.task_list`` against the local vLLM server and print the scores."""
    from evalscope import TaskConfig, run_task

    api_url = f"http://127.0.0.1:{config.vllm_port}/v1/chat/completions"
    gen_configs = config.generation_configs()
    repeats_per_task = config.repeats_per_task()

    all_results = {}
    valid_tasks = [ds for ds in config.task_list if ds in config.dataset_configs]
    per_ds_limit = (config.limit // len(valid_tasks)) if (config.limit != -1 and valid_tasks) else config.limit

    for ds_name in config.task_list:
        if ds_name not in config.dataset_configs:
            print(f"[WARN] Unknown dataset '{ds_name}', skipping.")
            continue

        entry = config.dataset_configs[ds_name]
        gen_config = gen_configs[entry["gen"]]
        repeats = repeats_per_task[ds_name]

        print(
            f"[INFO] Running dataset={ds_name}, repeats={repeats}, "
            f"max_tokens={gen_config['max_tokens']}, limit={per_ds_limit}"
        )

        task_kwargs = dict(
            model=config.model_type,
            api_url=api_url,
            eval_type="openai_api",
            repeats=repeats,
            datasets=[ds_name],
            dataset_args={ds_name: dict(entry["args"])},
            generation_config=gen_config,
            eval_batch_size=config.eval_batch_size,
            seed=config.seed,
            timeout=60000,
            stream=True,
        )
        if config.limit != -1:
            task_kwargs["limit"] = per_ds_limit
        result = run_task(task_cfg=TaskConfig(**task_kwargs))
        all_results[ds_name] = result

        if isinstance(result, dict):
            for bench_name, report in result.items():
                score = getattr(report, "score", None)
                print(f"[SCORE] {ds_name} ({bench_name}): {score}")

    print("[SUMMARY] Evaluation Results:")
    for ds_name, result in all_results.items():
        if isinstance(result, dict):
            for _, report in result.items():
                score = getattr(report, "score", "N/A")
                print(f"  {ds_name}: score={score}")
        else:
            print(f"  {ds_name}: {result}")
