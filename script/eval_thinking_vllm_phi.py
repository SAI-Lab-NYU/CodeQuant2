"""
Evaluate a Phi reasoning model through vLLM's OpenAI-compatible API and evalscope.

The tokenizer saved next to a clustered Phi model is not loadable by vLLM, so
pass the original Hugging Face snapshot directory with --tokenizer-path.

Usage:
  python eval_thinking_vllm_phi.py --model-dir <model_dir> --clean-tmp-dir <tmp_dir> --tokenizer-path <hf_snapshot_dir>
  python eval_thinking_vllm_phi.py --model-dir <model_dir> --clean-tmp-dir <tmp_dir> --tokenizer-path <hf_snapshot_dir> --no-quantization
  python eval_thinking_vllm_phi.py --model-dir <model_dir> --clean-tmp-dir <tmp_dir> --tokenizer-path <hf_snapshot_dir> --server-only
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.evaluation_utils import VllmEvalConfig, run_eval_cli


CONFIG = VllmEvalConfig(
    model_dir=None,
    model_type="phi",
    act_quant_bits=4,
    task_list=[
        "math_500",
        "gsm8k",
        "aime25",
        "aime26",
        "humaneval",
        "live_code_bench",
        "arc",
        "piqa",
        "winogrande",
    ],
    math_gen_config={
        "max_tokens": 4096,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    },
    aime_gen_config={
        "max_tokens": 4096,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    },
    mcq_gen_config={
        "max_tokens": 4096,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    },
    vllm_port=8802,
    clean_tmp_default=None,
    tokenizer_class_policy="remove",
)


if __name__ == "__main__":
    run_eval_cli(CONFIG)
