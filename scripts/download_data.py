import os
import sys
from pathlib import Path
from shutil import copy2
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.data_utils import save_processed_local_dataset


def download_raw_dataset_files(ds_name, target_root):
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    try:
        files = list_repo_files(repo_id=ds_name, repo_type='dataset')
    except Exception as exc:
        print(f"Unable to list raw files for {ds_name}: {exc}")
        return

    for file_name in files:
        if file_name.startswith('.'):
            continue
        try:
            print(f"Downloading raw file {file_name}...")
            downloaded_path = hf_hub_download(repo_id=ds_name, filename=file_name, repo_type='dataset')
            destination = target_root / Path(file_name).name
            copy2(downloaded_path, destination)
        except Exception as exc:
            print(f"Failed to download raw file {file_name} for {ds_name}: {exc}")

    print(f"Raw files downloaded for {ds_name} to {target_root}\n")


def download_datasets(dataset_names, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for ds_name in dataset_names:
        target = output_root / ds_name.replace('/', '_')
        target.mkdir(parents=True, exist_ok=True)

        try:
            print(f"Loading {ds_name}...")
            ds = load_dataset(ds_name, split='train')
            print(f"Saving to {target}...")
            ds.save_to_disk(str(target))
            save_processed_local_dataset(ds_name, output_root)
            print(f"Downloaded {ds_name} ({len(ds)} samples)\n")
        except Exception as e:
            print(f"Failed to load {ds_name}: {e}\n")
            print(f"Falling back to raw file download for {ds_name}...")
            download_raw_dataset_files(ds_name, target / 'raw_files')
            try:
                save_processed_local_dataset(ds_name, output_root)
                print(f"Normalized raw files for {ds_name} into local dataset format.\n")
            except Exception as normalize_exc:
                print(f"Normalization skipped for {ds_name}: {normalize_exc}\n")


if __name__ == '__main__':
    datasets = [
        'vnu-llm2023-ftdata/8k_crawl_web_uet',
        'vnu-llm2023-ftdata/1700_du_lieu_quy_che_DT',
        'vnu-llm2023-ftdata/500_tuyen_sinh_chinh_sua',
        'vnu-llm2023-ftdata/1597_out_hus_qa_final',
        'vnu-llm2023-ftdata/1k_finetune_and_200_hus',
        'vnu-llm2023-ftdata/620_sampled_QA_TVTS',
    ]
    output_root = Path(__file__).resolve().parents[1] / 'data' / 'raw'
    download_datasets(datasets, output_root)
