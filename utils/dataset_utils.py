from datasets import Dataset, load_dataset


def load_calibration_dataset(dataset_name: str, size: int, seed: int = 42) -> Dataset:
    """Load a shuffled subset of a dataset's train split with a single ``prompt`` column.

    Args:
        dataset_name: Hugging Face dataset id whose rows have a ``prompt`` or ``text`` column.
        size: Number of rows to keep after shuffling.
        seed: Shuffle seed.

    Returns:
        A ``datasets.Dataset`` whose only column is ``prompt``.

    Raises:
        ValueError: If the dataset has neither a ``prompt`` nor a ``text`` column.
    """
    dataset = load_dataset(dataset_name, split="train")

    if "prompt" in dataset.column_names:
        prompt_col = "prompt"
    elif "text" in dataset.column_names:
        prompt_col = "text"
    else:
        raise ValueError(f"No prompt-like column found: {dataset.column_names}")

    remove_columns = [name for name in dataset.column_names if name != prompt_col]
    dataset = dataset.map(lambda ex: {"prompt": ex[prompt_col]}, remove_columns=remove_columns)
    if prompt_col != "prompt":
        dataset = dataset.remove_columns([prompt_col])

    dataset = dataset.shuffle(seed=seed)
    size = min(size, len(dataset))
    return dataset.select(range(size))
