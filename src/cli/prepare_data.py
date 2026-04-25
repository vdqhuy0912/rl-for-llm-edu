#!/usr/bin/env python3
"""Prepare local project splits from downloaded Hugging Face datasets."""

import json
from pathlib import Path

from datasets import Dataset

from src.utils.data_utils import (
    RAW_DATA_ROOT,
    dataset_name_to_local_dir,
    load_local_dataset,
    normalize_qa_example,
    save_processed_local_dataset,
)


SPLITS_ROOT = Path("data/splits")

SFT_DATASET = "vnu-llm2023-ftdata/qa-daotao-sft"
RL_DATASET = "vnu-llm2023-ftdata/qa-daotao-cho-rl"
DATASETS = [SFT_DATASET, RL_DATASET]
REQUIRED_SPLITS = {
    SFT_DATASET: ("train", "validation", "test"),
    RL_DATASET: ("train", "validation", "test"),
}


def summarize_dataset(dataset_name: str, split: str) -> dict:
    dataset = load_local_dataset(dataset_name, split=split)
    sample_size = min(200, len(dataset))
    q_lengths = []
    a_lengths = []
    c_lengths = []

    for index in range(sample_size):
        row = normalize_qa_example(dataset[index])
        q_lengths.append(len(row["question"]))
        a_lengths.append(len(row["answer"]))
        c_lengths.append(len(row["context"]))

    return {
        "dataset": dataset_name,
        "split": split,
        "rows": len(dataset),
        "sampled_rows": sample_size,
        "avg_question_chars": round(sum(q_lengths) / sample_size, 1) if sample_size else 0,
        "avg_answer_chars": round(sum(a_lengths) / sample_size, 1) if sample_size else 0,
        "avg_context_chars": round(sum(c_lengths) / sample_size, 1) if sample_size else 0,
        "max_context_chars": max(c_lengths) if c_lengths else 0,
    }


def annotate_source(dataset: Dataset, dataset_name: str) -> Dataset:
    return dataset.map(lambda _: {"source_dataset": dataset_name})


def load_available_splits(dataset_name: str) -> dict[str, Dataset]:
    try:
        save_processed_local_dataset(dataset_name)
    except Exception as exc:
        print(f"Raw-file normalization skipped for {dataset_name}: {exc}")

    split_map = {}
    for split in ("train", "validation", "test"):
        try:
            split_map[split] = load_local_dataset(dataset_name, split=split)
        except Exception as exc:
            print(f"Skipping {dataset_name}:{split}: {exc}")
    return split_map


def validate_required_splits(available: dict[str, dict[str, Dataset]]) -> None:
    missing = []
    for dataset_name, required_splits in REQUIRED_SPLITS.items():
        present = available.get(dataset_name, {})
        for split in required_splits:
            if split not in present:
                missing.append(f"{dataset_name}:{split}")
    if missing:
        raise RuntimeError("Missing required dataset splits: " + ", ".join(missing))


def ensure_local_datasets() -> dict[str, dict[str, Dataset]]:
    available = {}

    print(f"Raw data root: {RAW_DATA_ROOT}")
    for dataset_name in DATASETS:
        target_dir = dataset_name_to_local_dir(dataset_name)
        print(f"Processing {dataset_name} -> {target_dir}")
        split_map = load_available_splits(dataset_name)
        for split in split_map:
            summary = summarize_dataset(dataset_name, split)
            print(json.dumps(summary, ensure_ascii=False))
        if split_map:
            available[dataset_name] = split_map

    return available


def save_split(dataset: Dataset, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    dataset.save_to_disk(str(output_dir))


def build_sft_splits(available: dict[str, dict[str, Dataset]]) -> dict:
    sft_dataset = available.get(SFT_DATASET, {})
    sft_train_source = sft_dataset.get("train")
    sft_val_source = sft_dataset.get("validation")
    sft_test_source = sft_dataset.get("test")

    if sft_train_source is None or sft_val_source is None or sft_test_source is None:
        sft_train = Dataset.from_list([])
        sft_val = Dataset.from_list([])
        sft_test = Dataset.from_list([])
    else:
        sft_train = annotate_source(sft_train_source, SFT_DATASET)
        sft_val = annotate_source(sft_val_source, SFT_DATASET)
        sft_test = annotate_source(sft_test_source, SFT_DATASET)

    save_split(sft_train, SPLITS_ROOT / "sft_train")
    save_split(sft_val, SPLITS_ROOT / "sft_val")
    save_split(sft_test, SPLITS_ROOT / "test_only")

    return {
        "train_rows": len(sft_train),
        "val_rows": len(sft_val),
        "test_rows": len(sft_test),
        "source_dataset": SFT_DATASET,
    }


def build_kto_splits(available: dict[str, dict[str, Dataset]]) -> dict:
    rl_dataset = available.get(RL_DATASET, {})
    rl_train_source = rl_dataset.get("train")
    rl_val_source = rl_dataset.get("validation")
    rl_test_source = rl_dataset.get("test")

    if rl_train_source is None or rl_val_source is None or rl_test_source is None:
        kto_train = Dataset.from_list([])
        kto_val = Dataset.from_list([])
        kto_test = Dataset.from_list([])
    else:
        kto_train = annotate_source(rl_train_source, RL_DATASET)
        kto_val = annotate_source(rl_val_source, RL_DATASET)
        kto_test = annotate_source(rl_test_source, RL_DATASET)

    save_split(kto_train, SPLITS_ROOT / "kto_train")
    save_split(kto_val, SPLITS_ROOT / "kto_val")
    save_split(kto_test, SPLITS_ROOT / "kto_test")

    return {
        "train_rows": len(kto_train),
        "val_rows": len(kto_val),
        "test_rows": len(kto_test),
        "train_source": RL_DATASET,
        "test_source": RL_DATASET,
    }


def save_manifest(manifest: dict) -> None:
    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = SPLITS_ROOT / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    available = ensure_local_datasets()
    validate_required_splits(available)
    sft_summary = build_sft_splits(available)
    kto_summary = build_kto_splits(available)

    manifest = {
        "rules": {
            "sft_dataset": SFT_DATASET,
            "rl_dataset": RL_DATASET,
            "sft_train_source_split": "train",
            "sft_val_source_split": "validation",
            "test_only_source_split": "test",
            "kto_train_source_split": "train",
            "kto_val_source_split": "validation",
            "kto_test_source_split": "test",
        },
        "available_datasets": {
            dataset_name: sorted(split_map.keys()) for dataset_name, split_map in available.items()
        },
        "sft": sft_summary,
        "kto": kto_summary,
    }

    save_manifest(manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
