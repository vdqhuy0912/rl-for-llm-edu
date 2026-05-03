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

### Xếp job vào tmux và tự đợi GPU rảnh
```bash
./scripts/tmux_wait_gpu.sh rl-sft ./scripts/workflow.sh train-sft
./scripts/tmux_wait_gpu.sh rl-kto ./scripts/workflow.sh train-kto
```

Có thể chỉnh ngưỡng trước khi chạy:
```bash
GPU_MAX_MEMORY_MB=500 GPU_MAX_UTILIZATION=5 GPU_WAIT_INTERVAL_SEC=30 \
./scripts/tmux_wait_gpu.sh rl-sft ./scripts/workflow.sh train-sft
```

Behavior:
- Script tạo tmux session detached.
- Mỗi `GPU_WAIT_INTERVAL_SEC` giây, script kiểm tra `nvidia-smi`.
- GPU đầu tiên có `memory.used <= GPU_MAX_MEMORY_MB` và `utilization.gpu <= GPU_MAX_UTILIZATION` sẽ được chọn.
- Command được chạy với `CUDA_VISIBLE_DEVICES=<gpu_index>` trong session đó.

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
./scripts/workflow.sh infer-model ./models/sft_checkpoints/final ./results/sft_infer test_only
./scripts/workflow.sh judge-file ./results/sft_infer/generated_responses.json ./results/sft_judge

```

### Chạy trực tiếp Python modules
```bash
python3 -m src.cli.run_sft
python3 -m src.cli.run_kto
python3 -m src.cli.run_infer --model-path ./models/kto_checkpoints/final --results-dir ./results/kto_infer --split-name kto_test
python3 -m src.cli.run_judge --input-path ./results/kto_infer/generated_responses.json --results-dir ./results/kto_judge
```

## Cấu hình

Các file cấu hình trong thư mục `configs/`:
- `sft_config.yaml`: Cấu hình SFT training
- `kto_config.yaml`: Cấu hình KTO training
- `eval_config.yaml`: Cấu hình evaluation

`system_prompt` được cấu hình trong cả ba file trên tại khóa `prompt.system_prompt`.
Prompt này được đưa vào format huấn luyện SFT, dữ liệu KTO và inference trước khi gửi câu trả lời sang LLM-as-a-judge.

### LoRA / QLoRA cho SFT

SFT luôn train bằng LoRA adapter. Chế độ LoRA hay QLoRA được chọn bằng `configs/sft_config.yaml`:

```yaml
qlora:
  load_in_4bit: false  # LoRA thường
```

Đổi sang QLoRA 4-bit:

```yaml
qlora:
  load_in_4bit: true
```

LoRA SFT hiện dùng:

```yaml
lora:
  r: 64
  lora_alpha: 128
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  lora_dropout: 0.1
  bias: "none"
  task_type: "CAUSAL_LM"
```

Khi `load_in_4bit: true`, code load model bằng `BitsAndBytesConfig` và gọi `prepare_model_for_kbit_training()` trước khi gắn LoRA adapter.

### LoRA / QLoRA cho KTO

KTO hỗ trợ ba chế độ trong `configs/kto_config.yaml`:

```yaml
tuning:
  mode: "qlora"  # options: "qlora", "lora", "none"
```

- `qlora`: load base/SFT model 4-bit và train LoRA adapter. Đây là mặc định khuyến nghị cho Qwen 8B.
- `lora`: load model ở fp16 và train LoRA adapter.
- `none`: full fine-tune, thường không phù hợp với GPU 48GB vì KTO cần reference policy.

LoRA KTO đã được đồng bộ với SFT:

```yaml
lora:
  r: 64
  lora_alpha: 128
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  lora_dropout: 0.1
  bias: "none"
  task_type: "CAUSAL_LM"
```

Lưu ý KTO của TRL yêu cầu actual batch size > 1, nên `per_device_train_batch_size` và `per_device_eval_batch_size` mặc định là `2`.

Chạy KTO mặc định bằng QLoRA:

```bash
./scripts/train_kto.sh
```

Ép chạy LoRA fp16:

```bash
./scripts/train_kto.sh --tuning-mode lora
```

Smoke test KTO bằng base model:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m src.cli.run_kto \
  --model-path Qwen/Qwen3-8B \
  --output-dir models/kto_qlora_smoke_test \
  --max-steps 1 \
  --num-train-samples 2 \
  --num-eval-samples 2 \
  --max-length 128 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --tuning-mode qlora
```

### Smoke test evaluation

`run_infer.py` hỗ trợ giới hạn số mẫu và số token sinh để test nhanh:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m src.cli.run_infer \
  --model-path Qwen/Qwen3-8B \
  --results-dir results/infer_smoke_test \
  --split-name test_only \
  --num-samples 1 \
  --max-new-tokens 32
```

Gemini judge hiện dùng model `gemini-3.1-flash-lite-preview` trong `configs/eval_config.yaml`.

## Yêu cầu hệ thống

- Python 3.10+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 100GB+ disk space cho models và data

## Lưu ý

- Python logic hiện nằm trong `src/cli/`; thư mục `scripts/` chỉ còn shell automation.
- `load_in_4bit: true` hoặc `optim: paged_adamw_8bit` đều cần `bitsandbytes>=0.46.1`.
- SFT chọn LoRA/QLoRA bằng `configs/sft_config.yaml -> qlora.load_in_4bit`.
- KTO chọn QLoRA/LoRA bằng `configs/kto_config.yaml -> tuning.mode`.
- Training logs được lưu trong `logs/`.
- Checkpoints trung gian được lưu theo `save_steps` của Hugging Face/TRL trong `models/sft_checkpoints/` và `models/kto_checkpoints/`.
- Final models được lưu ở:
  - `models/sft_checkpoints/final`
  - `models/kto_checkpoints/final`
- Inference và judging mặc định được lưu trong `results/`, hoặc thư mục bạn truyền vào `workflow.sh infer-model` và `workflow.sh judge-file`.
- Nếu `./scripts/eval_*.sh` được dùng, script sẽ tự load `.env`.

## Kết quả

Kết quả evaluation sẽ được lưu trong thư mục `results/`:
- `evaluation_results.json`: Chi tiết đánh giá
- `evaluation_results.csv`: Bảng tổng hợp
- `generated_responses.json`: Output inference trước khi judge
- `generated_responses.csv`: Bảng inference trước khi judge

## License

[LICENSE](LICENSE)
