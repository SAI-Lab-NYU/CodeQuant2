"""
Evaluate a Qwen3 thinking model through vLLM's OpenAI-compatible API and evalscope.

Usage:
  python eval_thinking_vllm.py --model-dir <model_dir> --clean-tmp-dir <tmp_dir>
  python eval_thinking_vllm.py --model-dir <model_dir> --clean-tmp-dir <tmp_dir> --no-quantization
  python eval_thinking_vllm.py --model-dir <model_dir> --clean-tmp-dir <tmp_dir> --server-only
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.evaluation_utils import VllmEvalConfig, run_eval_cli


THINKING_EXTRA = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}

CONFIG = VllmEvalConfig(
    model_dir=None,
    model_type="qwen3",
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
        "max_tokens": 30720,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        **THINKING_EXTRA,
    },
    aime_gen_config={
        "max_tokens": 30720,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        **THINKING_EXTRA,
    },
    mcq_gen_config={
        "max_tokens": 28672,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        **THINKING_EXTRA,
    },
    vllm_port=8801,
    clean_tmp_default=None,
    tokenizer_class_policy="replace",
)


if __name__ == "__main__":
    run_eval_cli(CONFIG)
