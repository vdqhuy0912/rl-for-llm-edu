#!/usr/bin/env python3
"""Download configured datasets into data/raw with local normalization fallback."""

from pathlib import Path
from shutil import copy2

from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

from src.utils.data_utils import save_processed_local_dataset


DATASETS = [
    "vnu-llm2023-ftdata/8k_crawl_web_uet",
    "vnu-llm2023-ftdata/1700_du_lieu_quy_che_DT",
    "vnu-llm2023-ftdata/500_tuyen_sinh_chinh_sua",
    "vnu-llm2023-ftdata/1597_out_hus_qa_final",
    "vnu-llm2023-ftdata/1k_finetune_and_200_hus",
    "vnu-llm2023-ftdata/620_sampled_QA_TVTS",
]


def download_raw_dataset_files(ds_name, target_root):
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    try:
        files = list_repo_files(repo_id=ds_name, repo_type="dataset")
    except Exception as exc:
        print(f"Unable to list raw files for {ds_name}: {exc}")
        return

    for file_name in files:
        if file_name.startswith("."):
            continue
        try:
            print(f"Downloading raw file {file_name}...")
            downloaded_path = hf_hub_download(repo_id=ds_name, filename=file_name, repo_type="dataset")
            destination = target_root / Path(file_name).name
            copy2(downloaded_path, destination)
        except Exception as exc:
            print(f"Failed to download raw file {file_name} for {ds_name}: {exc}")

    print(f"Raw files downloaded for {ds_name} to {target_root}\n")


def download_datasets(dataset_names, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for ds_name in dataset_names:
        target = output_root / ds_name.replace("/", "_")
        target.mkdir(parents=True, exist_ok=True)

        try:
            print(f"Loading {ds_name}...")
            dataset = load_dataset(ds_name, split="train")
            print(f"Saving to {target}...")
            dataset.save_to_disk(str(target))
            save_processed_local_dataset(ds_name, output_root)
            print(f"Downloaded {ds_name} ({len(dataset)} samples)\n")
        except Exception as exc:
            print(f"Failed to load {ds_name}: {exc}\n")
            print(f"Falling back to raw file download for {ds_name}...")
            download_raw_dataset_files(ds_name, target / "raw_files")
            try:
                save_processed_local_dataset(ds_name, output_root)
                print(f"Normalized raw files for {ds_name} into local dataset format.\n")
            except Exception as normalize_exc:
                print(f"Normalization skipped for {ds_name}: {normalize_exc}\n")


def main():
    output_root = Path("data/raw")
    download_datasets(DATASETS, output_root)


if __name__ == "__main__":
    main()
