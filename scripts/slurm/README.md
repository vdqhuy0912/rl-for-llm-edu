# Slurm job scripts

Bộ file này dùng để sửa/chạy nhanh trên server Slurm của bạn. Partition hiện đã đặt là `defq`.

## File có sẵn

- `00_test_gpu.sbatch`: kiểm tra job có được cấp GPU không, in `nvidia-smi` và trạng thái CUDA của PyTorch.
- `01_download_data.sbatch`: chạy `bash scripts/workflow.sh download-data`.
- `02_prepare_data.sbatch`: chạy `bash scripts/workflow.sh prepare-data`.
- `03_train_sft.sbatch`: train SFT, cần 1 GPU.
- `04_infer_sft.sbatch`: inference SFT final trên `test_only`, cần 1 GPU.
- `05_judge_sft.sbatch`: judge output SFT bằng Gemini API, không xin GPU.
- `06_train_kto.sbatch`: train KTO, cần 1 GPU.
- `07_infer_kto.sbatch`: inference KTO final trên `kto_test`, cần 1 GPU.
- `08_judge_kto.sbatch`: judge output KTO bằng Gemini API, không xin GPU.
- `09_custom_workflow.sbatch`: chạy nhanh một command bất kỳ của `scripts/workflow.sh`.

## Chạy trên server

Từ repo root trên server:

```bash
cd ~/vqhuy/rl-for-llm-edu
mkdir -p logs/slurm
```

Test GPU trước:

```bash
sbatch scripts/slurm/00_test_gpu.sbatch
squeue -u "$USER"
tail -f logs/slurm/rl-test-gpu-*.out
```

Chạy pipeline từng bước:

```bash
sbatch scripts/slurm/01_download_data.sbatch
sbatch scripts/slurm/02_prepare_data.sbatch
sbatch scripts/slurm/03_train_sft.sbatch
sbatch scripts/slurm/04_infer_sft.sbatch
sbatch scripts/slurm/05_judge_sft.sbatch
sbatch scripts/slurm/06_train_kto.sbatch
sbatch scripts/slurm/07_infer_kto.sbatch
sbatch scripts/slurm/08_judge_kto.sbatch
```

Chạy custom command:

```bash
sbatch --export=ALL,WORKFLOW_ARGS="infer-model ./models/sft_checkpoints/final ./results/tmp test_only 20" scripts/slurm/09_custom_workflow.sbatch
```

## Chỗ thường cần sửa

Trong mỗi file `.sbatch`, các dòng cần chỉnh nhanh nằm ở đầu file:

```bash
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
```

Ý nghĩa:

- `--partition=defq`: partition Slurm của server bạn.
- `--gres=gpu:1`: xin 1 GPU. Xóa dòng này cho job không cần GPU.
- `--cpus-per-task=8`: số CPU core cấp cho job.
- `--mem=80G`: RAM CPU.
- `--time=24:00:00`: thời gian tối đa.

Nếu muốn ép chạy vào node cụ thể:

```bash
#SBATCH --nodelist=node002
```

Chỉ thêm dòng này nếu bạn chắc node đó còn GPU và không bị admin drain/down.

## Nếu job không chạy

Xem lý do pending:

```bash
squeue -j <JOB_ID> -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"
```

Xem log:

```bash
tail -f logs/slurm/<JOB_NAME>-<JOB_ID>.out
tail -f logs/slurm/<JOB_NAME>-<JOB_ID>.err
```

Kiểm tra partition/node/GPU:

```bash
sinfo -O "Partition,NodeList,StateLong,Gres,GresUsed"
```

Kiểm tra conda trên server:

```bash
which conda
conda env list
```

Nếu `which conda` không có output hoặc environment không tên `rl-llm-edu`, cần sửa phần activate trong các file `.sbatch` hoặc tạo env theo `environment.yml`.
