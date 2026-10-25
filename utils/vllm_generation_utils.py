import atexit
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np


DEFAULT_TENSOR_PARALLEL = 2
DEFAULT_GPU_MEM_UTIL = 0.85
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_MAX_NUM_SEQS = 256
DEFAULT_BATCH_SIZE = 256
DEFAULT_VLLM_PORT = 8801

DEFAULT_GENERATION_CONFIG = {
    "max_tokens": 2048,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
}


def set_generation_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def model_type_from_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def default_reasoning_output_path(config: dict[str, Any]) -> str:
    model_type = model_type_from_name(config["model_name"])
    return os.path.join(config["path"]["rotation"], f"{model_type}_reasoning_gen.jsonl")


def resolve_generation_settings(config: dict[str, Any], args) -> dict[str, Any]:
    evaluation_config = config.get("evaluation", {})
    vllm_config = dict(evaluation_config.get("vllm", {}))
    generation_config = dict(DEFAULT_GENERATION_CONFIG)
    generation_config.update(evaluation_config.get("generation", {}))

    if args.gpu_mem_util is not None:
        vllm_config["gpu_mem_util"] = args.gpu_mem_util

    limit = args.limit if args.limit is not None else config["rotation"]["max_sample"]
    output_path = args.output_path or default_reasoning_output_path(config)

    return {
        "dataset_name": config["rotation"].get("hf_dataset", "openai/gsm8k"),
        "dataset_config": config["rotation"].get("hf_dataset_config", "main"),
        "problem_column": config["rotation"].get("text_column", "question"),
        "limit": limit,
        "output_path": output_path,
        "port": args.vllm_port or vllm_config.get("port", DEFAULT_VLLM_PORT),
        "vllm_config": vllm_config,
        "generation_config": generation_config,
    }


def resolve_model_dir(model_name: str, model_dir: Optional[str]) -> str:
    if model_dir:
        return model_dir

    from huggingface_hub import snapshot_download

    resolved = snapshot_download(repo_id=model_name)
    print(f"[INFO] Resolved model path: {resolved}")
    return resolved


def start_vllm_generation_server(model_dir: str, port: int, vllm_config: dict[str, Any]) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_dir,
        "--served-model-name",
        "model",
        "--port",
        str(port),
        "--max-model-len",
        str(vllm_config.get("max_model_len", DEFAULT_MAX_MODEL_LEN)),
        "--gpu-memory-utilization",
        str(vllm_config.get("gpu_mem_util", DEFAULT_GPU_MEM_UTIL)),
        "--tensor-parallel-size",
        str(vllm_config.get("tensor_parallel", DEFAULT_TENSOR_PARALLEL)),
        "--max-num-seqs",
        str(vllm_config.get("max_num_seqs", DEFAULT_MAX_NUM_SEQS)),
        "--dtype",
        "bfloat16",
        "--trust-remote-code",
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--reasoning-parser",
        "deepseek_r1",
    ]

    env = os.environ.copy()
    project_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")

    print(f"[INFO] Starting vLLM server on port {port}")
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)

    def cleanup() -> None:
        if proc.poll() is None:
            print("[INFO] Shutting down vLLM server...")
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(cleanup)
    return proc


def wait_for_generation_server(port: int, proc: subprocess.Popen, timeout: int = 1200) -> None:
    import urllib.request

    url = f"http://127.0.0.1:{port}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited early with code {proc.returncode}; check stderr for the real error"
            )
        try:
            urllib.request.urlopen(url, timeout=2)
            print(f"[INFO] vLLM server ready on port {port} (took {time.time() - start:.1f}s)")
            return
        except Exception:
            time.sleep(3)
    raise RuntimeError(f"vLLM server did not start within {timeout}s")


def load_generation_problems(dataset_name: str, limit: int, config_name: Optional[str] = None):
    from datasets import load_dataset

    dataset_dict = load_dataset(dataset_name, config_name) if config_name else load_dataset(dataset_name)
    split_name = list(dataset_dict.keys())[0]
    dataset = dataset_dict[split_name]
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    print(f"[INFO] Loaded {len(dataset)} problems from {dataset_name} (split={split_name})")
    return dataset


def resolve_problem_column(dataset, problem_column: str) -> str:
    candidates = [problem_column, "question", "problem", "input", "text"]
    resolved = next((column for column in candidates if column in dataset.column_names), None)
    if resolved is None:
        raise KeyError(f"No question column found. Available: {dataset.column_names}")
    if resolved != problem_column:
        print(f"[WARN] Column '{problem_column}' not found, using '{resolved}' instead")
    return resolved


def build_reasoning_prompts(dataset, tokenizer, problem_column: str) -> list[str]:
    prompts = []
    for item in dataset:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": item[problem_column]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append(prompt)
    return prompts


def run_reasoning_generation(
    port: int,
    prompts: list[str],
    generation_config: dict[str, Any],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[str]:
    from openai import OpenAI
    from tqdm import tqdm

    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="dummy")
    outputs = []
    with tqdm(total=len(prompts), desc="Generating", unit="sample") as pbar:
        for start in range(0, len(prompts), batch_size):
            for prompt in prompts[start:start + batch_size]:
                response = client.completions.create(
                    model="model",
                    prompt=prompt,
                    max_tokens=generation_config.get("max_tokens", DEFAULT_GENERATION_CONFIG["max_tokens"]),
                    temperature=generation_config.get("temperature", DEFAULT_GENERATION_CONFIG["temperature"]),
                    top_p=generation_config.get("top_p", DEFAULT_GENERATION_CONFIG["top_p"]),
                    extra_body={"top_k": generation_config.get("top_k", DEFAULT_GENERATION_CONFIG["top_k"])},
                )
                outputs.append(response.choices[0].text)
                pbar.update(1)
    return outputs


def save_reasoning_results(dataset, outputs: list[str], output_path: str, problem_column: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample_index, (item, text) in enumerate(zip(dataset, outputs)):
            record = {
                "index": sample_index,
                "input": item[problem_column],
                "target": item.get("answer", None),
                "prediction": text,
            }
            if "solution" in item:
                record["reference_solution"] = item["solution"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved {len(outputs)} results to {output_path}")
