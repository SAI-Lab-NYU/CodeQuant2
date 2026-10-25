import json
import random
from typing import List

from torch.utils.data import Dataset


class GenAnswerDataset(Dataset):
    """Reasoning traces written by ``script/gen_vllm.py``, one JSON record per line."""

    def __init__(
        self,
        jsonl_path: str,
        limit: int = -1,
        skip_empty_prediction: bool = False,
    ):
        self.records = []
        skipped = 0
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if skip_empty_prediction and not record.get("prediction", "").strip():
                    skipped += 1
                    continue
                self.records.append(record)
        if skipped:
            print(f"[INFO] GenAnswerDataset: skipped {skipped} records with empty prediction")
        if limit > 0:
            self.records = self.records[:limit]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = dict(self.records[idx])
        record.setdefault("source", "gsm8k")
        record.setdefault("text", "")
        return record


class WikiTextDataset(Dataset):
    """Non-empty rows of a WikiText-style dataset split as general-text calibration samples."""

    def __init__(
        self,
        dataset_name: str = "Self-GRIT/wikitext-2-raw-v1-preprocessed",
        split: str = "train",
        limit: int = -1,
    ):
        from datasets import load_dataset as hf_load

        dataset = hf_load(dataset_name, split=split)
        self.texts = [row["text"] for row in dataset if row["text"].strip()]
        if limit > 0:
            self.texts = self.texts[:limit]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "source": "wiki2",
            "text": self.texts[idx],
            "index": -1,
            "input": "",
            "target": "",
            "prediction": "",
        }


class MixedDataset(Dataset):
    """Mix generated answers and WikiText-2 at a configurable ratio."""

    def __init__(
        self,
        gen_dataset: Dataset,
        wiki_dataset: Dataset,
        total: int,
        wiki_ratio: float = 0.5,
        seed: int = 42,
        primary_label: str = "gen",
    ):
        random_generator = random.Random(seed)
        wiki_count = min(int(total * wiki_ratio), len(wiki_dataset))
        generated_count = min(total - wiki_count, len(gen_dataset))
        if generated_count + wiki_count < total:
            print(
                f"[WARN] MixedDataset: requested {total} samples but only "
                f"{generated_count + wiki_count} available "
                f"({generated_count} {primary_label} + {wiki_count} wiki)"
            )
        generated_indices = random_generator.sample(range(len(gen_dataset)), generated_count)
        wiki_indices = random_generator.sample(range(len(wiki_dataset)), wiki_count)
        records = []
        for sample_index in generated_indices:
            record = dict(gen_dataset[sample_index])
            record["source"] = "gsm8k"
            record.setdefault("text", "")
            records.append(record)
        for sample_index in wiki_indices:
            records.append(dict(wiki_dataset[sample_index]))
        random_generator.shuffle(records)
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def format_reasoning_batch(batch: dict, tokenizer) -> List[str]:
    """Render a collated batch into chat-templated strings; WikiText rows pass through as raw text."""
    return [
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": prediction},
            ],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        ) if source == "gsm8k" else text
        for input_text, prediction, source, text in zip(
            batch["input"],
            batch["prediction"],
            batch["source"],
            batch["text"],
        )
    ]
