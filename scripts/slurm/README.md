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
- `activate_sft_env.sh`: helper được các job source để dùng env `~/.conda/envs/SFT`.
- `11_judged_kto_pipeline.sbatch`: chạy full pipeline train SFT -> judge signal -> train KTO -> eval suite -> charts/report artifacts.
- `12_gold_positive_kto.sbatch`: sau khi có judge output, build thêm KTO data có `reference_answer` gán `label=True`, train `models/kto_gold_checkpoints`, eval và cập nhật report.

## Chạy trên server

Từ repo root trên server:

```bash
cd ~/vqhuy/rl-for-llm-edu
module load slurm/slurm/21.08.8
mkdir -p logs/slurm
```

Truy cập/test trực tiếp trên `node002` bằng `srun`:

```bash
srun --partition=defq --nodelist=node002 --gres=gpu:1 --pty bash
```

Trong shell `srun`, active env SFT:

```bash
cd ~/vqhuy/rl-for-llm-edu
source scripts/slurm/activate_sft_env.sh
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
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

Chạy full pipeline judged-KTO từ đầu:

```bash
sbatch scripts/slurm/11_judged_kto_pipeline.sbatch
```

Có thể giới hạn số mẫu để smoke test nhanh:

```bash
sbatch --export=ALL,EVAL_NUM_SAMPLES=48,KTO_SIGNAL_TRAIN_SAMPLES=256,KTO_SIGNAL_VAL_SAMPLES=128 scripts/slurm/11_judged_kto_pipeline.sbatch
```

Chạy thêm KTO phiên bản có gold answer positive sau full pipeline. Nếu muốn chạy tự động sau job full pipeline hiện tại:

```bash
sbatch --dependency=afterok:52792 --export=ALL,SOURCE_RUN_NAME=20260514_181257 scripts/slurm/12_gold_positive_kto.sbatch
```

Nếu không truyền `SOURCE_RUN_NAME`, script sẽ tự lấy run mới nhất có đủ:

```bash
results/judged_kto_train_<RUN_NAME>/judge/evaluation_results.json
```

Chạy custom command:

```bash
sbatch --export=ALL,WORKFLOW_ARGS="infer-model ./models/sft_checkpoints/final ./results/tmp test_only 20" scripts/slurm/09_custom_workflow.sbatch
```

Các job mặc định dùng Python trong:

```bash
~/.conda/envs/SFT
```

Nếu cần dùng env khác, truyền `SFT_ENV_PATH` khi submit:

```bash
sbatch --export=ALL,SFT_ENV_PATH=/path/to/env scripts/slurm/00_test_gpu.sbatch
```

## Chỗ thường cần sửa

Trong mỗi file `.sbatch`, các dòng cần chỉnh nhanh nằm ở đầu file:

```bash
#SBATCH --partition=defq
#SBATCH --nodelist=node002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
```

Ý nghĩa:

- `--partition=defq`: partition Slurm của server bạn.
- `--nodelist=node002`: ép job chạy trên node A100 đang dùng.
- `--gres=gpu:1`: xin 1 GPU. Xóa dòng này cho job không cần GPU.
- `--cpus-per-task=8`: số CPU core cấp cho job.
- `--mem=80G`: RAM CPU.
- `--time=24:00:00`: thời gian tối đa.

Nếu muốn đổi node:

```bash
#SBATCH --nodelist=<node-name>
```

Chỉ đổi dòng này nếu bạn chắc node đó còn GPU và không bị admin drain/down.

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

Kiểm tra Python mà job sẽ dùng:

```bash
bash scripts/slurm/activate_sft_env.sh
python -c "import sys; print(sys.executable)"
```

Nếu helper báo không tìm thấy env, sửa `SFT_ENV_PATH` hoặc tạo lại env theo `environment.yml`.
