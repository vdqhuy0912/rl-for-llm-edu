#!/usr/bin/env python3
"""Normalize project datasets in data/raw, then build deterministic train/val/test splits."""

import argparse
import json
from pathlib import Path

from datasets import Dataset, concatenate_datasets

from src.utils.data_utils import (
    RAW_DATA_ROOT,
    dataset_name_to_local_dir,
    load_local_dataset,
    normalize_qa_example,
    save_processed_local_dataset,
)


SEED = 42
DEFAULT_TRAIN_VAL_RATIO = 0.9
TVTS_TRAIN_RATIO = 0.5
SPLITS_ROOT = Path("data/splits")

TRAIN_ONLY_DATASETS = [
    "vnu-llm2023-ftdata/8k_crawl_web_uet",
    "vnu-llm2023-ftdata/1700_du_lieu_quy_che_DT",
    "vnu-llm2023-ftdata/500_tuyen_sinh_chinh_sua",
    "vnu-llm2023-ftdata/1597_out_hus_qa_final",
]
TEST_ONLY_DATASET = "vnu-llm2023-ftdata/1k_finetune_and_200_hus"
TVTS_DATASET = "vnu-llm2023-ftdata/620_sampled_QA_TVTS"
DATASETS = [*TRAIN_ONLY_DATASETS, TEST_ONLY_DATASET, TVTS_DATASET]


def summarize_dataset(dataset_name: str) -> dict:
    dataset = load_local_dataset(dataset_name)
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
        "rows": len(dataset),
        "sampled_rows": sample_size,
        "avg_question_chars": round(sum(q_lengths) / sample_size, 1) if sample_size else 0,
        "avg_answer_chars": round(sum(a_lengths) / sample_size, 1) if sample_size else 0,
        "avg_context_chars": round(sum(c_lengths) / sample_size, 1) if sample_size else 0,
        "max_context_chars": max(c_lengths) if c_lengths else 0,
    }


def split_train_val(dataset: Dataset, train_ratio: float, seed: int) -> tuple[Dataset, Dataset]:
    split = dataset.train_test_split(test_size=1 - train_ratio, seed=seed, shuffle=True)
    return split["train"], split["test"]


def annotate_source(dataset: Dataset, dataset_name: str) -> Dataset:
    return dataset.map(lambda _: {"source_dataset": dataset_name})


def ensure_local_datasets() -> dict[str, Dataset]:
    available = {}

    print(f"Raw data root: {RAW_DATA_ROOT}")
    for dataset_name in DATASETS:
        target_dir = dataset_name_to_local_dir(dataset_name)
        print(f"Processing {dataset_name} -> {target_dir}")
        try:
            save_processed_local_dataset(dataset_name)
            summary = summarize_dataset(dataset_name)
            print(json.dumps(summary, ensure_ascii=False))
            available[dataset_name] = load_local_dataset(dataset_name)
        except Exception as exc:
            print(f"Skipping {dataset_name}: {exc}")

    return available


def save_split(dataset: Dataset, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    dataset.save_to_disk(str(output_dir))


def build_sft_splits(available: dict[str, Dataset]) -> dict:
    train_parts = []
    val_parts = []

    for offset, dataset_name in enumerate(TRAIN_ONLY_DATASETS):
        dataset = available.get(dataset_name)
        if dataset is None:
            continue
        annotated = annotate_source(dataset, dataset_name)
        train_split, val_split = split_train_val(
            annotated,
            train_ratio=DEFAULT_TRAIN_VAL_RATIO,
            seed=SEED + offset,
        )
        train_parts.append(train_split)
        val_parts.append(val_split)

    tvts_dataset = available.get(TVTS_DATASET)
    if tvts_dataset is not None:
        annotated_tvts = annotate_source(tvts_dataset, TVTS_DATASET)
        tvts_train, tvts_val = split_train_val(
            annotated_tvts,
            train_ratio=TVTS_TRAIN_RATIO,
            seed=SEED + 100,
        )
        train_parts.append(tvts_train)
        val_parts.append(tvts_val)
    else:
        tvts_train = None
        tvts_val = None

    sft_train = concatenate_datasets(train_parts) if train_parts else Dataset.from_list([])
    sft_val = concatenate_datasets(val_parts) if val_parts else Dataset.from_list([])
    sft_train = sft_train.shuffle(seed=SEED)
    sft_val = sft_val.shuffle(seed=SEED)

    test_dataset = available.get(TEST_ONLY_DATASET)
    sft_test = annotate_source(test_dataset, TEST_ONLY_DATASET) if test_dataset is not None else Dataset.from_list([])

    save_split(sft_train, SPLITS_ROOT / "sft_train")
    save_split(sft_val, SPLITS_ROOT / "sft_val")
    save_split(sft_test, SPLITS_ROOT / "test_only")

    return {
        "train_rows": len(sft_train),
        "val_rows": len(sft_val),
        "test_rows": len(sft_test),
        "tvts_train_rows": len(tvts_train) if tvts_train is not None else 0,
        "tvts_val_rows": len(tvts_val) if tvts_val is not None else 0,
    }


def build_kto_splits(available: dict[str, Dataset]) -> dict:
    tvts_dataset = available.get(TVTS_DATASET)
    if tvts_dataset is None:
        kto_train = Dataset.from_list([])
        kto_val = Dataset.from_list([])
    else:
        annotated_tvts = annotate_source(tvts_dataset, TVTS_DATASET)
        kto_train, kto_val = split_train_val(
            annotated_tvts,
            train_ratio=TVTS_TRAIN_RATIO,
            seed=SEED + 100,
        )

    save_split(kto_train, SPLITS_ROOT / "kto_train")
    save_split(kto_val, SPLITS_ROOT / "kto_val")

    return {
        "train_rows": len(kto_train),
        "val_rows": len(kto_val),
    }


def save_manifest(manifest: dict) -> None:
    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = SPLITS_ROOT / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalize-only", action="store_true", help="Only normalize/check local raw datasets.")
    parser.add_argument("--split-only", action="store_true", help="Only create deterministic splits.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.normalize_only and args.split_only:
        raise ValueError("Choose at most one of --normalize-only or --split-only.")

    if args.split_only:
        available = {}
        for dataset_name in DATASETS:
            try:
                available[dataset_name] = load_local_dataset(dataset_name)
            except Exception as exc:
                print(f"Skipping {dataset_name}: {exc}")
    else:
        available = ensure_local_datasets()

    if args.normalize_only:
        return

    sft_summary = build_sft_splits(available)
    kto_summary = build_kto_splits(available)

    manifest = {
        "seed": SEED,
        "rules": {
            "test_only_dataset": TEST_ONLY_DATASET,
            "tvts_dataset": TVTS_DATASET,
            "tvts_train_ratio": TVTS_TRAIN_RATIO,
            "default_train_val_ratio_for_other_train_sets": DEFAULT_TRAIN_VAL_RATIO,
        },
        "available_datasets": sorted(available.keys()),
        "sft": sft_summary,
        "kto": kto_summary,
    }

    save_manifest(manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
