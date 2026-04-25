# RL for LLM Education

Dự án training Reinforcement Learning cho Large Language Models trong lĩnh vực giáo dục Việt Nam.

## Tổng quan

Dự án này thực hiện fine-tuning model Qwen 3 8B theo pipeline:
1. **Supervised Fine-Tuning (SFT)** với LoRA/QLoRA
2. **Knowledge Transfer Optimization (KTO)** từ model SFT
3. **Đánh giá** bằng LLM-as-a-judge với Google Gemini

## Cấu trúc dự án

```
rl-for-llm-edu/
├── src/
│   ├── cli/              # Python entrypoints cho download/process/train/eval
│   └── utils/            # Utility functions
├── configs/              # Configuration files
├── data/
│   ├── raw/              # Raw downloaded data
│   └── splits/           # Train/val/test splits
├── models/
│   ├── sft_checkpoints/  # SFT model checkpoints
│   └── kto_checkpoints/  # KTO model checkpoints
├── scripts/              # Shell automation scripts
├── prompt/               # Judge prompts cho Gemini
├── docs/                 # Ghi chú thiết kế / labeling rules
├── results/              # Evaluation results
├── logs/                 # Training logs
├── notebooks/            # Jupyter notebooks
├── requirements.txt      # Python dependencies
├── environment.yml       # Conda environment
└── README.md
```

## Nguồn dữ liệu

### Training Data
- `vnu-llm2023-ftdata/8k_crawl_web_uet`: 8k samples crawled web data
- `vnu-llm2023-ftdata/1700_du_lieu_quy_che_DT`: 1700 legal/regulatory data
- `vnu-llm2023-ftdata/500_tuyen_sinh_chinh_sua`: 500 admission data
- `vnu-llm2023-ftdata/1597_out_hus_qa_final`: 1597 HUS Q&A data

### Test Data
- `vnu-llm2023-ftdata/1k_finetune_and_200_hus`: 1k finetune + 200 HUS samples

### Mixed Data
- `vnu-llm2023-ftdata/620_sampled_QA_TVTS`: 620 sampled Q&A dùng để tách train/val cho KTO và bổ sung train/val cho SFT

## Quy tắc chia dữ liệu

- `vnu-llm2023-ftdata/1k_finetune_and_200_hus` chỉ dùng làm test set.
- `vnu-llm2023-ftdata/620_sampled_QA_TVTS` được chia cố định `50/50` cho train và val.
- Các dataset train còn lại được chia cố định `90/10` cho train và val.
- Tất cả split dùng seed cố định nên chạy lại vẫn ra cùng kết quả.

Chạy bước chuẩn hóa và tạo split:
```bash
./scripts/preprocess_data.sh
./scripts/split_data.sh
```

Kết quả được lưu trong `data/splits/`:
- `sft_train`
- `sft_val`
- `kto_train`
- `kto_val`
- `test_only`
- `split_manifest.json`

## Cài đặt

### 1. Tạo môi trường Conda
```bash
conda env create -f environment.yml
conda activate rl-llm-edu
```

### 2. Hoặc cài đặt với pip
```bash
pip install -r requirements.txt
```

### 3. Thiết lập API keys
```bash
echo 'GEMINI_API_KEY=your_gemini_api_key_here' > .env
```

## Sử dụng

### Cách 1: Chạy full pipeline
```bash
./scripts/full_pipeline.sh
```

### Cách 2: Chạy từng bước bằng shell scripts
```bash
./scripts/download_data.sh
./scripts/preprocess_data.sh
./scripts/split_data.sh
./scripts/train_sft.sh
./scripts/eval_sft.sh
./scripts/train_kto.sh
./scripts/eval_kto.sh
```

### Cách 3: Dùng dispatcher duy nhất
```bash
./scripts/workflow.sh download-data
./scripts/workflow.sh preprocess-data
./scripts/workflow.sh split-data
./scripts/workflow.sh train-sft
./scripts/workflow.sh eval-sft
./scripts/workflow.sh train-kto
./scripts/workflow.sh eval-kto
```

### Đánh giá một model bất kỳ
```bash
./scripts/eval_model.sh ./models/sft_checkpoints/final ./results/sft_eval
```

### Chạy trực tiếp Python modules
```bash
python3 -m src.cli.run_sft
python3 -m src.cli.run_kto
python3 -m src.cli.run_eval --model-path ./models/kto_checkpoints/final --results-dir ./results/kto_eval
```

## Cấu hình

Các file cấu hình trong thư mục `configs/`:
- `sft_config.yaml`: Cấu hình SFT training
- `kto_config.yaml`: Cấu hình KTO training
- `eval_config.yaml`: Cấu hình evaluation

## Yêu cầu hệ thống

- Python 3.10+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 100GB+ disk space cho models và data

## Lưu ý

- Python logic hiện nằm trong `src/cli/`; thư mục `scripts/` chỉ còn shell automation.
- Training logs được lưu trong `logs/`.
- Checkpoints trung gian được lưu theo `save_steps` của Hugging Face/TRL trong `models/sft_checkpoints/` và `models/kto_checkpoints/`.
- Final models được lưu ở:
  - `models/sft_checkpoints/final`
  - `models/kto_checkpoints/final`
- Evaluation results mặc định được lưu trong `results/`, hoặc thư mục bạn truyền vào `eval_model.sh`.
- Nếu `./scripts/eval_*.sh` được dùng, script sẽ tự load `.env`.

## Kết quả

Kết quả evaluation sẽ được lưu trong thư mục `results/`:
- `evaluation_results.json`: Chi tiết đánh giá
- `evaluation_results.csv`: Bảng tổng hợp

## License

[LICENSE](LICENSE)
