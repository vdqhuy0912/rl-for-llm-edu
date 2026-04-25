#!/usr/bin/env python3
"""
Normalize project datasets in data/raw and print quick stats for training.
"""

import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_utils import (
    RAW_DATA_ROOT,
    dataset_name_to_local_dir,
    load_local_dataset,
    normalize_qa_example,
    save_processed_local_dataset,
)


DATASETS = [
    "vnu-llm2023-ftdata/8k_crawl_web_uet",
    "vnu-llm2023-ftdata/1700_du_lieu_quy_che_DT",
    "vnu-llm2023-ftdata/500_tuyen_sinh_chinh_sua",
    "vnu-llm2023-ftdata/1597_out_hus_qa_final",
    "vnu-llm2023-ftdata/1k_finetune_and_200_hus",
    "vnu-llm2023-ftdata/620_sampled_QA_TVTS",
]


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


def main():
    print(f"Raw data root: {RAW_DATA_ROOT}")
    for dataset_name in DATASETS:
        target_dir = dataset_name_to_local_dir(dataset_name)
        print(f"Processing {dataset_name} -> {target_dir}")
        save_processed_local_dataset(dataset_name)
        summary = summarize_dataset(dataset_name)
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
