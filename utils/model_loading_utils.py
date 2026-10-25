from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_causal_lm(model_name: str, tie: bool, **model_kwargs: Any):
    config = AutoConfig.from_pretrained(model_name, tie_word_embeddings=False) if tie else AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    if tie:
        retie_lm_head(model)

    return config, model, tokenizer


def retie_lm_head(model) -> None:
    embed_tokens = model.model.embed_tokens
    lm_head = model.lm_head
    lm_head_device = lm_head.weight.device
    with torch.no_grad():
        lm_head.weight.data = embed_tokens.weight.data.clone().to(
            dtype=lm_head.weight.dtype,
            device=lm_head_device,
        )
        if lm_head.bias is not None:
            lm_head.bias.data = embed_tokens.bias.data.to(
                dtype=lm_head.weight.dtype,
                device=lm_head_device,
            )
