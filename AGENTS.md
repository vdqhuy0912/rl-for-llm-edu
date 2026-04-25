# AGENTS.md - Tóm tắt công việc đã thực hiện và các quyết định đã chốt

## Dự án: RL for LLM Education - Training Loop Qwen 3 8B

### Cập nhật gần nhất: 25/04/2026

## Tổng quan nhiệm vụ
Thiết kế và setup dự án training loop cho model Qwen 3 8B với pipeline:
1. Supervised Fine-Tuning (SFT) với LoRA/QLoRA
2. Knowledge Transfer Optimization (KTO)
3. Evaluation với LLM-as-a-judge (Gemini)

## Các tác vụ đã hoàn thành

### ✅ 1. Phân tích yêu cầu và thiết kế kiến trúc
- **Input**: Yêu cầu thiết kế training loop 3 bước với data sources cụ thể
- **Output**: Kiến trúc project hoàn chỉnh với cấu trúc thư mục logic
- **Decision**: Chọn cấu trúc modular với src/, configs/, scripts/ tách biệt

### ✅ 2. Tạo cấu trúc thư mục
- **Created directories**:
  - `src/sft/`, `src/kto/`, `src/evaluation/`, `src/utils/`
  - `configs/`, `models/sft_checkpoints/`, `models/kto_checkpoints/`
  - `scripts/`, `notebooks/`, `results/`
  - `data/raw/`, `data/processed/`, `data/splits/`
- **Rationale**: Tổ chức code theo modules và phases để dễ maintain

### ✅ 3. Setup dependencies và environment
- **requirements.txt**: 15+ packages bao gồm transformers, peft, trl, google-generativeai
- **environment.yml**: Conda environment với CUDA support
- **Considerations**: Chọn versions tương thích, include dev tools (black, isort, flake8)

### ✅ 4. Tạo configuration files
- **sft_config.yaml**: LoRA/QLoRA params, training hyperparams, data paths
- **kto_config.yaml**: KTO-specific params (beta=0.1), learning rate thấp hơn
- **eval_config.yaml**: Gemini API config, judge prompts, evaluation metrics
- **Design choice**: YAML format cho dễ edit, structured theo phases

### ✅ 5. Implement utility modules
- **model_utils.py**: Load/save models, setup logging, checkpoint management
- **data_utils.py**: HF dataset loading, preprocessing cho SFT/KTO, data formatting
- **Architecture**: Modular functions, error handling, logging integration

### ✅ 6. Tạo training scripts
- **run_sft.py**: Complete SFT pipeline với LoRA/QLoRA, data loading, training loop
- **run_kto.py**: KTO training từ SFT checkpoint, TRL integration
- **run_eval.py**: Response generation + Gemini evaluation pipeline
- **Features**: Progress logging, error handling, result saving

### ✅ 7. Documentation và README
- **README.md**: Comprehensive guide với setup instructions, usage examples
- **Structure**: Overview, data sources, installation, usage steps, requirements
- **Language**: Vietnamese để phù hợp với project context

### ✅ 8. Session memory tracking
- **project_plan.md**: Detailed project plan với progress tracking
- **Scope**: User session memory để maintain context across interactions

### ✅ 9. Xác nhận lại yêu cầu với user và cập nhật assumptions
- **Base model ưu tiên**: `Qwen/Qwen3-8B`
- **Candidate nếu cần kiểm tra KTO compatibility**: `Qwen/Qwen3.5-9B`
- **Judge prompt chính thức**: `prompt/PROMPT-LLM-as-a-Judge.md`
- **Yêu cầu mới**: Thiết kế rõ bước chuyển đổi dữ liệu cho KTO thay vì giả định sẵn preferred/dispreferred pairs

### ✅ 10. Chuẩn hóa lại dữ liệu và thiết kế bước chuyển đổi KTO
- **Observed schema thực tế**:
  - Hầu hết dataset dùng `question`, `answer`, `reference`
  - `620_sampled_QA_TVTS` dùng `references` và cờ `insufficial context` (lệch chính tả so với các dataset khác)
- **Normalization strategy**:
  - Chuẩn hóa về các field chung: `question`, `answer`, `context`, `insufficient_context`, `multi_intent`
  - Dùng chung prompt format:
    - `### Câu hỏi`
    - `### Ngữ cảnh tham chiếu`
    - `### Trả lời`
- **KTO conversion design**:
  - Positive sample: dùng `answer` gốc làm `completion`, `label=True`
  - Negative sample: sinh tổng hợp có kiểm soát, `label=False`
  - Thứ tự chiến lược negative:
    1. `partial_answer_for_multi_intent`
    2. `raw_context_dump`
    3. `overconfident_placeholder`
    4. `generic_non_answer`
- **Preview tool**: thêm `scripts/preview_kto_data.py` để xem trực tiếp output conversion

### ✅ 11. Cập nhật evaluation config theo prompt judge thực tế
- **Removed assumption**: rubric cũ `Relevance / Accuracy / Helpfulness` 1-5 không còn là judge chính
- **New behavior**:
  - Đọc prompt từ `prompt/prompts.py`
  - Classify `(Q, C)` trước theo `CLASS_1/CLASS_2/CLASS_3`
  - Nếu `CLASS_1` dùng `PROMPT_QA_CLASS1`
  - Nếu `CLASS_2` dùng `PROMPT_QA_CLASS2`
  - File prompt hiện có `PROMPT_QA_CLASSIFIER`, `PROMPT_QA_CLASS1`, `PROMPT_QA_CLASS2`, `PROMPT_QA_CLASS3`

### ✅ 12. Chốt rule split dữ liệu cố định và lưu vào `data/splits/`
- **Test-only dataset**: `vnu-llm2023-ftdata/1k_finetune_and_200_hus`
- **TVTS split rule**: `vnu-llm2023-ftdata/620_sampled_QA_TVTS` chia cố định `50/50` cho train/val
- **Other train datasets**: mỗi dataset còn lại chia cố định `90/10` cho train/val
- **Determinism**: dùng seed `42` và offset cố định theo dataset để lần nào chạy lại cũng ra cùng split
- **Artifacts**:
  - `data/splits/sft_train`
  - `data/splits/sft_val`
  - `data/splits/kto_train`
  - `data/splits/kto_val`
  - `data/splits/test_only`
  - `data/splits/split_manifest.json`

### ✅ 13. Thêm orchestration scripts cho toàn bộ pipeline
- **Shell entrypoints**:
  - `scripts/download_data.sh`
  - `scripts/preprocess_data.sh`
  - `scripts/split_data.sh`
  - `scripts/train_sft.sh`
  - `scripts/train_kto.sh`
  - `scripts/eval_sft.sh`
  - `scripts/eval_kto.sh`
  - `scripts/eval_all.sh`
  - `scripts/full_pipeline.sh`
- **Evaluation flexibility**:
  - `scripts/run_eval.py` nhận `--model-path`
  - `scripts/run_eval.py` nhận `--results-dir`
  - shell eval scripts tự load `.env` qua `scripts/common.sh`

### ✅ 14. Refactor Python entrypoints vào `src/`
- **New package**: `src/cli/`
- **Moved main logic**:
  - `src/cli/download_data.py`
  - `src/cli/process_data.py`
  - `src/cli/preview_kto_data.py`
  - `src/cli/run_sft.py`
  - `src/cli/run_kto.py`
  - `src/cli/run_eval.py`
- **Compatibility strategy**:
  - giữ `scripts/*.py` làm wrapper mỏng để không phá shell scripts và command cũ
- **Logging/checkpoint updates**:
  - log file theo logger name trong `logs/`
  - tiếp tục lưu checkpoint theo `save_steps` của Hugging Face/TRL trong `models/...`
  - lưu thêm final model ở `models/sft_checkpoints/final` và `models/kto_checkpoints/final`
- **Progress bars**:
  - training loops dùng progress bar mặc định của `Trainer`/`KTOTrainer`
  - explicit `disable_tqdm=False` được set mặc định trong code nếu config không override

## Technical decisions made

### Model & Framework Choices
- **Qwen/Qwen3-8B**: Selected as default base model
- **Qwen/Qwen3.5-9B**: Candidate model nếu cần kiểm tra hỗ trợ KTO tốt hơn
- **LoRA/QLoRA**: For efficient fine-tuning với limited GPU memory
- **TRL Library**: For KTO implementation (battle-tested)
- **Transformers**: Standard for model loading/training

### Data Pipeline
- **HuggingFace Datasets**:
  - Train:
    - `vnu-llm2023-ftdata/8k_crawl_web_uet`
    - `vnu-llm2023-ftdata/1700_du_lieu_quy_che_DT`
    - `vnu-llm2023-ftdata/500_tuyen_sinh_chinh_sua`
    - `vnu-llm2023-ftdata/1597_out_hus_qa_final`
  - Test:
    - `vnu-llm2023-ftdata/1k_finetune_and_200_hus`
  - Auxiliary split / KTO source:
    - `vnu-llm2023-ftdata/620_sampled_QA_TVTS`
- **Preprocessing**: Tokenization + unified QA/context formatting
- **KTO Format**: Không còn giả định có sẵn pair; chuyển đổi từ QA sang positive/negative rows bằng heuristic có ghi metadata

### Evaluation Strategy
- **Gemini API**: LLM-as-judge cho automated evaluation
- **Judge source**: `prompt/prompts.py`
- **Flow**: classifier prompt → chọn prompt đánh giá theo class
- **Output**: JSON + CSV results với classification và judge output thô

### Execution Layout
- **Source of truth for Python code**: `src/`
- **Wrapper layer**: `scripts/*.py` only delegates to `src/cli/*`
- **Automation layer**: `scripts/*.sh` orchestrates download → preprocess → split → train → eval

## Potential challenges identified
- **Network/data access**: Một số dataset chưa được download local đầy đủ trong workspace hiện tại
- **API rate limits**: Gemini evaluation may hit rate limits với large datasets
- **Environment gap**: một số local env có thể chưa cài package `datasets`, nên `process_data.py` chưa chạy được ngay nếu chưa setup env

## Next steps recommended
1. **Environment setup**: Run `conda env create -f environment.yml`
2. **KTO preview**: Run `python scripts/preview_kto_data.py` để xem conversion
3. **Run deterministic split build**: `python3 scripts/process_data.py`
4. **Config tuning**: Adjust hyperparameters based on hardware
5. **Testing**: Run small-scale tests trước full training

## Files created/modified
- 15+ directories created
- Core files đã được cập nhật thêm theo các quyết định mới:
  - `configs/sft_config.yaml`
  - `configs/kto_config.yaml`
  - `configs/eval_config.yaml`
  - `src/utils/data_utils.py`
  - `src/utils/model_utils.py`
  - `src/cli/*.py`
  - `scripts/*.sh`
  - `scripts/run_eval.py`
  - `scripts/preview_kto_data.py`
- README.md updated
- project_plan.md created in session memory
