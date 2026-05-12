# Chạy pipeline bằng Slurm

File này gom các lệnh cần dùng khi chạy project trên server Slurm mới.

Các placeholder cần sửa theo cluster:

- `<PARTITION_GPU>`: tên partition GPU, ví dụ `gpu`, `a100`, `h100`.
- `<ACCOUNT>`: account/project Slurm nếu server yêu cầu. Nếu không cần, bỏ dòng `#SBATCH --account=...`.
- `<REPO>`: thư mục repo trên server, ví dụ `$HOME/rl-for-llm-edu`.
- `<CONDA_BASE>`: đường dẫn conda, thường là `$HOME/miniconda3` hoặc `$HOME/anaconda3`.

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

### 3.10. Full pipeline trong một job

Chỉ dùng nếu server cho phép job rất dài. Cách ổn định hơn là dùng dependency chain ở mục 5.

```bash
cat > scripts/slurm/full_pipeline.sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=rl-full
#SBATCH --partition=<PARTITION_GPU>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd <REPO>
source <CONDA_BASE>/etc/profile.d/conda.sh
conda activate rl-llm-edu

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/workflow.sh full-pipeline
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

Full pipeline một job:

```bash
sbatch scripts/slurm/full_pipeline.sbatch
```

## 5. Submit theo dependency chain

Cách này đảm bảo bước sau chỉ chạy nếu bước trước thành công.

```bash
cd <REPO>

J_DOWNLOAD=$(sbatch --parsable scripts/slurm/download_data.sbatch)
J_PREPARE=$(sbatch --parsable --dependency=afterok:${J_DOWNLOAD} scripts/slurm/prepare_data.sbatch)
J_SFT=$(sbatch --parsable --dependency=afterok:${J_PREPARE} scripts/slurm/train_sft.sbatch)
J_INFER_SFT=$(sbatch --parsable --dependency=afterok:${J_SFT} scripts/slurm/infer_sft.sbatch)
J_JUDGE_SFT=$(sbatch --parsable --dependency=afterok:${J_INFER_SFT} scripts/slurm/judge_sft.sbatch)
J_KTO=$(sbatch --parsable --dependency=afterok:${J_JUDGE_SFT} scripts/slurm/train_kto.sbatch)
J_INFER_KTO=$(sbatch --parsable --dependency=afterok:${J_KTO} scripts/slurm/infer_kto.sbatch)
J_JUDGE_KTO=$(sbatch --parsable --dependency=afterok:${J_INFER_KTO} scripts/slurm/judge_kto.sbatch)

printf 'download=%s\nprepare=%s\nsft=%s\ninfer_sft=%s\njudge_sft=%s\nkto=%s\ninfer_kto=%s\njudge_kto=%s\n' \
  "$J_DOWNLOAD" "$J_PREPARE" "$J_SFT" "$J_INFER_SFT" "$J_JUDGE_SFT" "$J_KTO" "$J_INFER_KTO" "$J_JUDGE_KTO"
```

Nếu không cần judge SFT trước KTO, có thể cho KTO chạy ngay sau SFT:

```bash
cd <REPO>

J_DOWNLOAD=$(sbatch --parsable scripts/slurm/download_data.sbatch)
J_PREPARE=$(sbatch --parsable --dependency=afterok:${J_DOWNLOAD} scripts/slurm/prepare_data.sbatch)
J_SFT=$(sbatch --parsable --dependency=afterok:${J_PREPARE} scripts/slurm/train_sft.sbatch)
J_INFER_SFT=$(sbatch --parsable --dependency=afterok:${J_SFT} scripts/slurm/infer_sft.sbatch)
J_JUDGE_SFT=$(sbatch --parsable --dependency=afterok:${J_INFER_SFT} scripts/slurm/judge_sft.sbatch)
J_KTO=$(sbatch --parsable --dependency=afterok:${J_SFT} scripts/slurm/train_kto.sbatch)
J_INFER_KTO=$(sbatch --parsable --dependency=afterok:${J_KTO} scripts/slurm/infer_kto.sbatch)
J_JUDGE_KTO=$(sbatch --parsable --dependency=afterok:${J_INFER_KTO} scripts/slurm/judge_kto.sbatch)
```

## 6. Lệnh theo dõi và debug

Xem queue:

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
