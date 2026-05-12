# Chạy pipeline bằng Slurm

File này gom các lệnh cần dùng khi chạy project trên server Slurm mới.

Các placeholder cần sửa theo cluster:

- `<PARTITION_GPU>`: tên partition GPU, ví dụ `gpu`, `a100`, `h100`.
- `<ACCOUNT>`: account/project Slurm nếu server yêu cầu. Nếu không cần, bỏ dòng `#SBATCH --account=...`.
- `<REPO>`: thư mục repo trên server, ví dụ `$HOME/rl-for-llm-edu`.
- `<CONDA_BASE>`: đường dẫn conda, thường là `$HOME/miniconda3` hoặc `$HOME/anaconda3`.

## 0. Giải thích các thành phần lệnh

### Lệnh kiểm tra Slurm/GPU

```bash
sinfo
```

- `sinfo`: xem trạng thái các partition/node trong Slurm.
- Dùng để biết cluster có partition nào, node nào đang `idle`, `alloc`, `down`.

```bash
sinfo -o "%P %D %t %G"
```

- `sinfo`: xem thông tin node/partition.
- `-o`: chọn format output.
- `%P`: tên partition.
- `%D`: số node.
- `%t`: trạng thái node.
- `%G`: GPU/GRES có trên node, ví dụ `gpu:a100:4`.

```bash
squeue -u "$USER"
```

- `squeue`: xem danh sách job đang chờ hoặc đang chạy.
- `-u "$USER"`: chỉ xem job của user hiện tại.
- `"$USER"`: biến môi trường Linux chứa username hiện tại.

```bash
scontrol show partition
```

- `scontrol`: xem/điều khiển thông tin Slurm chi tiết.
- `show partition`: in cấu hình partition, gồm giới hạn thời gian, node, policy.

```bash
nvidia-smi
```

- Xem GPU, VRAM, driver CUDA, process đang dùng GPU.
- Trên một số server, lệnh này chỉ chạy được bên trong job đã xin GPU.

### Lệnh chuẩn bị môi trường

```bash
cd <REPO>
```

- `cd`: chuyển thư mục làm việc.
- `<REPO>`: thư mục chứa project này trên server.
- Mọi lệnh sau đó chạy tương đối từ repo root.

```bash
conda env create -f environment.yml
```

- `conda env create`: tạo conda environment mới.
- `-f environment.yml`: đọc danh sách package từ file `environment.yml`.
- Environment được tạo tên `rl-llm-edu` theo dòng `name:` trong file.

```bash
conda activate rl-llm-edu
```

- Kích hoạt environment `rl-llm-edu`.
- Sau khi activate, `python`, `pip`, `torch`, `transformers` sẽ lấy từ environment này.

```bash
pip install -U "bitsandbytes>=0.46.1"
```

- `pip install`: cài package Python.
- `-U`: nâng cấp nếu đã có bản cũ.
- `"bitsandbytes>=0.46.1"`: cần cho QLoRA 4-bit và optimizer `paged_adamw_8bit`.
- Dấu quote giúp shell không hiểu nhầm ký tự `>`.

```bash
pip install -r requirements.txt
```

- `-r requirements.txt`: cài tất cả package liệt kê trong file requirements.

```bash
mkdir -p logs/slurm results models data/raw data/splits
```

- `mkdir`: tạo thư mục.
- `-p`: không báo lỗi nếu thư mục đã tồn tại; tự tạo parent directory nếu thiếu.
- `logs/slurm`: log Slurm.
- `results`: kết quả inference/judge.
- `models`: checkpoint model.
- `data/raw`, `data/splits`: dữ liệu tải về và dữ liệu đã chuẩn hóa.

```bash
printf 'GEMINI_API_KEY=%s\n' 'YOUR_GEMINI_API_KEY' > .env
```

- `printf`: ghi text theo format.
- `GEMINI_API_KEY=%s\n`: format một dòng biến môi trường.
- `'YOUR_GEMINI_API_KEY'`: giá trị key cần thay bằng key thật.
- `> .env`: ghi output vào file `.env`, ghi đè file cũ nếu có.

```bash
chmod 600 .env
```

- `chmod`: đổi quyền file.
- `600`: chỉ owner được đọc/ghi; user khác không đọc được.
- Dùng để tránh lộ API key.

```bash
huggingface-cli login
```

- Đăng nhập Hugging Face CLI.
- Cần nếu dataset/model yêu cầu token hoặc muốn tránh rate limit.

### Cấu trúc file sbatch

```bash
cat > scripts/slurm/train_sft.sbatch <<'EOF'
...
EOF
```

- `cat`: đọc nội dung từ stdin và ghi ra stdout.
- `>`: redirect stdout vào file.
- `scripts/slurm/train_sft.sbatch`: file job Slurm sẽ được tạo.
- `<<'EOF'`: heredoc; mọi dòng sau đó được ghi nguyên văn cho đến dòng `EOF`.
- Dấu quote trong `'EOF'` giữ nguyên biến như `$USER`, không expand ngay lúc tạo file.

```bash
#!/usr/bin/env bash
```

- Shebang cho biết file sẽ chạy bằng `bash`.
- `/usr/bin/env bash` tìm `bash` theo `PATH`, portable hơn hard-code `/bin/bash`.

```bash
#SBATCH --job-name=rl-train-sft
```

- Đặt tên job trong Slurm.
- Tên này xuất hiện trong `squeue`, `sacct`, và tên log nếu dùng `%x`.

```bash
#SBATCH --partition=<PARTITION_GPU>
```

- Chọn partition để submit job.
- Partition thường tương ứng nhóm node, ví dụ GPU A100/H100 hoặc CPU.

```bash
#SBATCH --account=<ACCOUNT>
```

- Chỉ định account/project billing.
- Nếu cluster không dùng account, xóa dòng này.

```bash
#SBATCH --gres=gpu:1
```

- `gres`: generic resource.
- `gpu:1`: xin 1 GPU.
- Các job train/infer cần dòng này; job download/prepare/judge có thể không cần GPU.

```bash
#SBATCH --cpus-per-task=8
```

- Xin 8 CPU cores cho một task.
- Ảnh hưởng đến dataloader, preprocessing, tokenizer và một phần pipeline Python.

```bash
#SBATCH --mem=80G
```

- Xin 80GB RAM hệ thống.
- Đây là RAM CPU, không phải VRAM GPU.

```bash
#SBATCH --time=24:00:00
```

- Giới hạn thời gian job.
- Format là `HH:MM:SS` hoặc `D-HH:MM:SS`.
- Hết thời gian Slurm sẽ kill job.

```bash
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
```

- `--output`: file log stdout.
- `--error`: file log stderr.
- `%x`: tên job.
- `%j`: job id.

```bash
set -euo pipefail
```

- `-e`: lỗi là dừng script.
- `-u`: dùng biến chưa khai báo là lỗi.
- `-o pipefail`: pipeline lỗi nếu một command bên trong pipeline lỗi.
- Giúp job fail rõ ràng thay vì chạy tiếp với trạng thái sai.

```bash
source <CONDA_BASE>/etc/profile.d/conda.sh
```

- Load shell hook của conda.
- Cần để `conda activate` hoạt động trong non-interactive shell của Slurm.

```bash
export TOKENIZERS_PARALLELISM=false
```

- Tắt parallel warning của Hugging Face tokenizers.
- Giảm nguy cơ deadlock/cảnh báo khi dataloader fork process.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

- Cấu hình allocator CUDA của PyTorch.
- `expandable_segments:True` giúp giảm fragmentation VRAM trong một số workload train/infer.

### Lệnh chạy workflow của project

```bash
bash scripts/workflow.sh download-data
```

- Chạy module `src.cli.download_data`.
- Tải hoặc materialize dữ liệu nguồn.

```bash
bash scripts/workflow.sh prepare-data
```

- Chạy module `src.cli.prepare_data`.
- Chuẩn hóa dữ liệu và tạo các split cố định trong `data/splits/`.

```bash
bash scripts/workflow.sh train-sft
```

- Chạy module `src.cli.run_sft`.
- Train SFT từ `data/splits/sft_train`, eval bằng `sft_val`.
- Output chính: `models/sft_checkpoints/final`.

```bash
bash scripts/workflow.sh train-kto
```

- Chạy module `src.cli.run_kto`.
- Train KTO từ `data/splits/kto_train`, eval bằng `kto_val`.
- Output chính: `models/kto_checkpoints/final`.

```bash
bash scripts/workflow.sh infer-model ./models/sft_checkpoints/final ./results/sft_eval/inference test_only
```

- `infer-model`: generate response bằng một model bất kỳ.
- `./models/sft_checkpoints/final`: model path.
- `./results/sft_eval/inference`: thư mục lưu kết quả inference.
- `test_only`: split dùng để generate.
- Output chính: `generated_responses.json` và `generated_responses.csv`.

```bash
bash scripts/workflow.sh infer-model ./models/sft_checkpoints/final ./results/sft_eval_20/inference test_only 20
```

- Giống lệnh infer ở trên.
- `20`: chỉ chạy 20 samples, dùng để test nhanh.

```bash
bash scripts/workflow.sh judge-file ./results/sft_eval/inference/generated_responses.json ./results/sft_eval/judge
```

- `judge-file`: chấm file generated responses bằng Gemini.
- Tham số 1: file input từ inference.
- Tham số 2: thư mục output judge.
- Cần `GEMINI_API_KEY` trong `.env` hoặc environment.

### Lệnh submit và theo dõi job

```bash
sbatch scripts/slurm/train_sft.sbatch
```

- `sbatch`: submit file job vào Slurm.
- Slurm trả về job id, ví dụ `Submitted batch job 12345`.

```bash
tail -f logs/slurm/rl-train-sft-12345.out
```

- `tail`: xem cuối file.
- `-f`: follow, tiếp tục in log mới khi file được ghi thêm.
- Dùng để theo dõi training log realtime.

```bash
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES,MaxRSS
```

- `sacct`: xem lịch sử/accounting của job đã chạy.
- `-j <JOB_ID>`: chọn job id.
- `--format=...`: chọn cột hiển thị.
- `State`: trạng thái như `COMPLETED`, `FAILED`, `CANCELLED`, `TIMEOUT`.
- `ExitCode`: mã thoát, `0:0` thường là thành công.
- `Elapsed`: thời gian chạy.
- `AllocTRES`: tài nguyên được cấp.
- `MaxRSS`: RAM CPU tối đa đã dùng.

```bash
scancel <JOB_ID>
```

- Hủy một job đang chờ hoặc đang chạy.

```bash
srun --partition=<PARTITION_GPU> --account=<ACCOUNT> --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=01:00:00 --pty bash
```

- `srun`: chạy command tương tác qua Slurm.
- `--partition`, `--account`, `--gres`, `--cpus-per-task`, `--mem`, `--time`: xin tài nguyên giống `sbatch`.
- `--pty bash`: mở shell tương tác trên node được cấp.
- Dùng để debug môi trường, CUDA, package trước khi submit job dài.

## 1. Kiểm tra Slurm và GPU

```bash
sinfo
sinfo -o "%P %D %t %G"
squeue -u "$USER"
scontrol show partition
nvidia-smi
```

Nếu `nvidia-smi` chỉ chạy được trong job GPU, dùng job test ngắn ở mục 4.

## 2. Chuẩn bị repo và môi trường

```bash
cd <REPO>

conda env create -f environment.yml
conda activate rl-llm-edu

pip install -U "bitsandbytes>=0.46.1"
pip install -r requirements.txt

mkdir -p logs/slurm results models data/raw data/splits
```

Nếu đã có environment:

```bash
cd <REPO>
conda activate rl-llm-edu
pip install -U "bitsandbytes>=0.46.1"
pip install -r requirements.txt
mkdir -p logs/slurm results models data/raw data/splits
```

Thiết lập key Gemini để judge:

```bash
cd <REPO>
printf 'GEMINI_API_KEY=%s\n' 'YOUR_GEMINI_API_KEY' > .env
chmod 600 .env
```

Nếu dataset hoặc model Hugging Face cần token:

```bash
huggingface-cli login
```

## 3. Tạo các file sbatch

Tạo thư mục:

```bash
cd <REPO>
mkdir -p scripts/slurm logs/slurm
```

### 3.1. Job test GPU

```bash
cat > scripts/slurm/test_gpu.sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=rl-test-gpu
#SBATCH --partition=<PARTITION_GPU>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd <REPO>
source <CONDA_BASE>/etc/profile.d/conda.sh
conda activate rl-llm-edu

hostname
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
EOF
```

### 3.2. Download data

```bash
cat > scripts/slurm/download_data.sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=rl-download
#SBATCH --partition=<PARTITION_GPU>
#SBATCH --account=<ACCOUNT>
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd <REPO>
source <CONDA_BASE>/etc/profile.d/conda.sh
conda activate rl-llm-edu

bash scripts/workflow.sh download-data
EOF
```

### 3.3. Prepare data

```bash
cat > scripts/slurm/prepare_data.sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=rl-prepare
#SBATCH --partition=<PARTITION_GPU>
#SBATCH --account=<ACCOUNT>
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd <REPO>
source <CONDA_BASE>/etc/profile.d/conda.sh
conda activate rl-llm-edu

bash scripts/workflow.sh prepare-data
EOF
```

### 3.4. Train SFT

```bash
cat > scripts/slurm/train_sft.sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=rl-train-sft
#SBATCH --partition=<PARTITION_GPU>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd <REPO>
source <CONDA_BASE>/etc/profile.d/conda.sh
conda activate rl-llm-edu

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/workflow.sh train-sft
EOF
```

### 3.5. Infer SFT final

```bash
cat > scripts/slurm/infer_sft.sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=rl-infer-sft
#SBATCH --partition=<PARTITION_GPU>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd <REPO>
source <CONDA_BASE>/etc/profile.d/conda.sh
conda activate rl-llm-edu

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/workflow.sh infer-model ./models/sft_checkpoints/final ./results/sft_eval/inference test_only
EOF
```

### 3.6. Judge SFT final

```bash
cat > scripts/slurm/judge_sft.sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=rl-judge-sft
#SBATCH --partition=<PARTITION_GPU>
#SBATCH --account=<ACCOUNT>
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd <REPO>
source <CONDA_BASE>/etc/profile.d/conda.sh
conda activate rl-llm-edu

bash scripts/workflow.sh judge-file ./results/sft_eval/inference/generated_responses.json ./results/sft_eval/judge
EOF
```

### 3.7. Train KTO

```bash
cat > scripts/slurm/train_kto.sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=rl-train-kto
#SBATCH --partition=<PARTITION_GPU>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd <REPO>
source <CONDA_BASE>/etc/profile.d/conda.sh
conda activate rl-llm-edu

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/workflow.sh train-kto
EOF
```

### 3.8. Infer KTO final

```bash
cat > scripts/slurm/infer_kto.sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=rl-infer-kto
#SBATCH --partition=<PARTITION_GPU>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd <REPO>
source <CONDA_BASE>/etc/profile.d/conda.sh
conda activate rl-llm-edu

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/workflow.sh infer-model ./models/kto_checkpoints/final ./results/kto_eval/inference kto_test
EOF
```

### 3.9. Judge KTO final

```bash
cat > scripts/slurm/judge_kto.sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=rl-judge-kto
#SBATCH --partition=<PARTITION_GPU>
#SBATCH --account=<ACCOUNT>
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd <REPO>
source <CONDA_BASE>/etc/profile.d/conda.sh
conda activate rl-llm-edu

bash scripts/workflow.sh judge-file ./results/kto_eval/inference/generated_responses.json ./results/kto_eval/judge
EOF
```

## 4. Submit từng job

Test GPU:

```bash
sbatch scripts/slurm/test_gpu.sbatch
squeue -u "$USER"
tail -f logs/slurm/rl-test-gpu-*.out
```

Chạy từng bước thủ công:

```bash
sbatch scripts/slurm/download_data.sbatch
sbatch scripts/slurm/prepare_data.sbatch
sbatch scripts/slurm/train_sft.sbatch
sbatch scripts/slurm/infer_sft.sbatch
sbatch scripts/slurm/judge_sft.sbatch
sbatch scripts/slurm/train_kto.sbatch
sbatch scripts/slurm/infer_kto.sbatch
sbatch scripts/slurm/judge_kto.sbatch
```

## 5. Lệnh theo dõi và debug

Xem job:

```bash
squeue -u "$USER"
squeue -j <JOB_ID>
scontrol show job <JOB_ID>
```

Xem log:

```bash
tail -f logs/slurm/<JOB_NAME>-<JOB_ID>.out
tail -f logs/slurm/<JOB_NAME>-<JOB_ID>.err
less logs/slurm/<JOB_NAME>-<JOB_ID>.out
less logs/slurm/<JOB_NAME>-<JOB_ID>.err
```

Xem lịch sử job:

```bash
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES,MaxRSS
sacct -u "$USER" --starttime today
```

Hủy job:

```bash
scancel <JOB_ID>
scancel -u "$USER"
```

Chạy shell tương tác GPU để debug:

```bash
srun --partition=<PARTITION_GPU> --account=<ACCOUNT> --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=01:00:00 --pty bash
cd <REPO>
source <CONDA_BASE>/etc/profile.d/conda.sh
conda activate rl-llm-edu
nvidia-smi
```

## 6. Giữ tmux cho chạy ngoài Slurm

Nếu đang ở node cho phép chạy trực tiếp, vẫn có thể dùng tmux như cũ:

```bash
tmux new -s rl-train
bash scripts/workflow.sh train-sft
bash scripts/workflow.sh train-kto
```

Các lệnh tmux hay dùng:

```bash
tmux ls
tmux attach -t rl-train
tmux kill-session -t rl-train
```

## 7. Chạy inference/judge nhanh với ít mẫu

Tạo job infer SFT 20 mẫu:

```bash
cat > scripts/slurm/infer_sft_20.sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=rl-infer-sft-20
#SBATCH --partition=<PARTITION_GPU>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd <REPO>
source <CONDA_BASE>/etc/profile.d/conda.sh
conda activate rl-llm-edu

bash scripts/workflow.sh infer-model ./models/sft_checkpoints/final ./results/sft_eval_20/inference test_only 20
EOF

sbatch scripts/slurm/infer_sft_20.sbatch
```

Judge 20 mẫu:

```bash
cat > scripts/slurm/judge_sft_20.sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=rl-judge-sft-20
#SBATCH --partition=<PARTITION_GPU>
#SBATCH --account=<ACCOUNT>
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd <REPO>
source <CONDA_BASE>/etc/profile.d/conda.sh
conda activate rl-llm-edu

bash scripts/workflow.sh judge-file ./results/sft_eval_20/inference/generated_responses.json ./results/sft_eval_20/judge
EOF

sbatch scripts/slurm/judge_sft_20.sbatch
```

## 8. Output quan trọng

Sau SFT:

```bash
ls -lah models/sft_checkpoints/final
ls -lah results/sft_eval/inference/generated_responses.json
ls -lah results/sft_eval/judge/evaluation_results.json
```

Sau KTO:

```bash
ls -lah models/kto_checkpoints/final
ls -lah results/kto_eval/inference/generated_responses.json
ls -lah results/kto_eval/judge/evaluation_results.json
```

## 9. Ghi chú tài nguyên

- Qwen/Qwen3-8B nên chạy trên GPU VRAM lớn. Nếu OOM, giảm batch size trong `configs/sft_config.yaml` hoặc `configs/kto_config.yaml`.
- KTO mặc định đang dùng `tuning.mode: qlora` để giảm VRAM.
- Nếu cụm dùng module thay vì conda path trực tiếp, thay:

```bash
source <CONDA_BASE>/etc/profile.d/conda.sh
conda activate rl-llm-edu
```

bằng lệnh của cluster, ví dụ:

```bash
module load anaconda3
conda activate rl-llm-edu
```

- Nếu partition CPU không có GPU, có thể đổi các job `download_data`, `prepare_data`, `judge_*` sang partition CPU và bỏ dòng `#SBATCH --gres=gpu:1`.
