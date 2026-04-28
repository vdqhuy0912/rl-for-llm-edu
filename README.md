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

- `vnu-llm2023-ftdata/qa-daotao-sft`: dataset tổng hợp dùng cho SFT
- `vnu-llm2023-ftdata/qa-daotao-cho-rl`: dataset tổng hợp dùng cho RL/KTO/DPO

## Quy tắc chia dữ liệu

- Mỗi dataset trên Hugging Face đã có sẵn `train` / `validation` / `test`.
- `prepare_data.py` không tự chia lại dữ liệu nữa, mà mirror trực tiếp các split sẵn có vào `data/splits/`.
- `qa-daotao-sft/train` -> `sft_train`
- `qa-daotao-sft/validation` -> `sft_val`
- `qa-daotao-sft/test` -> `test_only`
- `qa-daotao-cho-rl/train` -> `kto_train`
- `qa-daotao-cho-rl/validation` -> `kto_val`
- `qa-daotao-cho-rl/test` -> `kto_test`

Chạy bước chuẩn hóa và tạo split:
```bash
./scripts/prepare_data.sh
```

Ý nghĩa:
- `prepare-data`: materialize, verify, mirror các split local từ `data/raw` vào `data/splits/`

Kết quả được lưu trong `data/splits/`:
- `sft_train`
- `sft_val`
- `kto_train`
- `kto_val`
- `kto_test`
- `test_only`
- `split_manifest.json`

`kto_train` được dùng để train KTO. `kto_test` được dùng khi chạy `eval-kto`.

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

Nếu dùng QLoRA 4-bit hoặc optimizer `paged_adamw_8bit`, cần có `bitsandbytes>=0.46.1`.
Nếu môi trường hiện tại đã tạo sẵn từ trước, chạy thêm:
```bash
pip install -U "bitsandbytes>=0.46.1"
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
./scripts/prepare_data.sh
./scripts/train_sft.sh
./scripts/eval_sft.sh
./scripts/train_kto.sh
./scripts/eval_kto.sh
```

### Chạy train/eval bằng tmux (khuyến nghị cho job dài)
```bash
# tạo session
tmux new -s rl-train

# trong tmux, chạy tuần tự
./scripts/workflow.sh train-sft
./scripts/workflow.sh eval-sft
./scripts/workflow.sh train-kto
./scripts/workflow.sh eval-kto
```

Các lệnh tmux cơ bản:
```bash
# tách khỏi session (detach): nhấn Ctrl+b rồi nhấn d

# xem danh sách session
tmux ls

# vào lại session
tmux attach -t rl-train

# đóng session khi hoàn tất
tmux kill-session -t rl-train
```

Nếu muốn vừa train vừa theo dõi log:
```bash
tmux new -s rl-train

# window 1
./scripts/workflow.sh train-sft

# tạo window mới: Ctrl+b rồi nhấn c
# window 2
tail -f logs/train_sft_*.log
```

### Cách 3: Dùng dispatcher duy nhất
```bash
./scripts/workflow.sh download-data
./scripts/workflow.sh prepare-data
./scripts/workflow.sh train-sft
./scripts/workflow.sh eval-sft
./scripts/workflow.sh train-kto
./scripts/workflow.sh eval-kto
```

### Đánh giá một model bất kỳ
```bash
./scripts/eval_model.sh ./models/sft_checkpoints/final ./results/sft_eval test_only
./scripts/eval_model.sh ./models/kto_checkpoints/final ./results/kto_eval kto_test
```

### Chạy trực tiếp Python modules
```bash
python3 -m src.cli.run_sft
python3 -m src.cli.run_kto
python3 -m src.cli.run_eval --model-path ./models/kto_checkpoints/final --results-dir ./results/kto_eval --split-name kto_test
```

## Cấu hình

Các file cấu hình trong thư mục `configs/`:
- `sft_config.yaml`: Cấu hình SFT training
- `kto_config.yaml`: Cấu hình KTO training
- `eval_config.yaml`: Cấu hình evaluation

`system_prompt` được cấu hình trong cả ba file trên tại khóa `prompt.system_prompt`.
Prompt này được đưa vào format huấn luyện SFT, dữ liệu KTO và inference trước khi gửi câu trả lời sang LLM-as-a-judge.

## Yêu cầu hệ thống

- Python 3.10+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 100GB+ disk space cho models và data

## Lưu ý

- Python logic hiện nằm trong `src/cli/`; thư mục `scripts/` chỉ còn shell automation.
- `load_in_4bit: true` hoặc `optim: paged_adamw_8bit` đều cần `bitsandbytes>=0.46.1`.
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
