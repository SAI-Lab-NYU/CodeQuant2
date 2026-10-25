import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.entrypoint_utils import load_yaml_config
from utils.vllm_generation_utils import (
    build_reasoning_prompts,
    load_generation_problems,
    resolve_generation_settings,
    resolve_model_dir,
    resolve_problem_column,
    run_reasoning_generation,
    save_reasoning_results,
    set_generation_seed,
    start_vllm_generation_server,
    wait_for_generation_server,
)


def parse_args():
    parser = argparse.ArgumentParser(description="generate reasoning traces for rotation calibration")
    parser.add_argument("--config", type=str, required=True, help="config file name or path")
    parser.add_argument("--model-dir", type=str, default=None, help="local model path; defaults to HF snapshot")
    parser.add_argument("--output-path", type=str, default=None, help="override output jsonl path")
    parser.add_argument("--limit", type=int, default=None, help="override rotation.max_sample")
    parser.add_argument("--vllm-port", type=int, default=None, help="vLLM server port")
    parser.add_argument("--gpu-mem-util", type=float, default=None, help="vLLM GPU memory utilization")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config(args.config)
    set_generation_seed(42)

    model_name = config["model_name"]
    settings = resolve_generation_settings(config, args)

    print(f"[INFO] Model:   {model_name}")
    print(
        f"[INFO] Dataset: {settings['dataset_name']} "
        f"(column={settings['problem_column']}, limit={settings['limit']})"
    )
    print(f"[INFO] Output:  {settings['output_path']}")
    print(f"[INFO] Port:    {settings['port']}")

    model_dir = resolve_model_dir(model_name, args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    dataset = load_generation_problems(
        settings["dataset_name"],
        settings["limit"],
        settings["dataset_config"],
    )
    problem_column = resolve_problem_column(dataset, settings["problem_column"])
    prompts = build_reasoning_prompts(dataset, tokenizer, problem_column)

    server_proc = start_vllm_generation_server(model_dir, settings["port"], settings["vllm_config"])
    wait_for_generation_server(settings["port"], server_proc)

    outputs = run_reasoning_generation(
        settings["port"],
        prompts,
        settings["generation_config"],
        batch_size=settings["vllm_config"].get("batch_size", 256),
    )
    save_reasoning_results(dataset, outputs, settings["output_path"], problem_column)
