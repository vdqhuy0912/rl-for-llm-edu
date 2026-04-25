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
│   ├── sft/              # SFT training code
│   ├── kto/              # KTO training code
│   ├── evaluation/       # Evaluation code
│   └── utils/            # Utility functions
├── configs/              # Configuration files
├── data/
│   ├── raw/              # Raw downloaded data
│   ├── processed/        # Processed data
│   └── splits/           # Train/val/test splits
├── models/
│   ├── sft_checkpoints/  # SFT model checkpoints
│   └── kto_checkpoints/  # KTO model checkpoints
├── scripts/              # Training scripts
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
- `vnu-llm2023-ftdata/620_sampled_QA_TVTS`: 620 sampled Q&A for train/test split

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
export GEMINI_API_KEY="your_gemini_api_key_here"
```

## Sử dụng

### Bước 1: Supervised Fine-Tuning (SFT)
```bash
python scripts/run_sft.py
```

### Bước 2: Knowledge Transfer Optimization (KTO)
```bash
python scripts/run_kto.py
```

### Bước 3: Đánh giá với Gemini
```bash
python scripts/run_eval.py
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

- Model Qwen 3 8B có thể cần điều chỉnh tên model trong config
- Data format có thể cần điều chỉnh dựa trên cấu trúc thực tế của datasets
- KTO yêu cầu data format đặc biệt (preferred/dispreferred pairs)

## Kết quả

Kết quả evaluation sẽ được lưu trong thư mục `results/`:
- `evaluation_results.json`: Chi tiết đánh giá
- `evaluation_results.csv`: Bảng tổng hợp

## License

[LICENSE](LICENSE)
