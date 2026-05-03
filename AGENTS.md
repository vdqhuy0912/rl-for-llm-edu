# AGENTS.md

## Dự án
RL for LLM Education: pipeline huấn luyện và đánh giá cho `Qwen/Qwen3-8B`.

## Trạng thái hiện tại
- Pipeline chính: `SFT -> KTO -> inference -> Gemini judge`
- Source code Python nằm trong `src/cli/` và `src/utils/`
- Shell orchestration nằm ở `scripts/workflow.sh`
- `final` model được lưu tại:
  - `models/sft_checkpoints/final`
  - `models/kto_checkpoints/final`

## Các quyết định còn hiệu lực

### 1. Model và tuning
- Base model mặc định: `Qwen/Qwen3-8B`
- SFT train bằng LoRA; có thể bật QLoRA qua `configs/sft_config.yaml -> qlora.load_in_4bit`
- KTO mặc định dùng `tuning.mode: qlora` để tránh OOM
- KTO có thể override sang `lora` hoặc `none`, nhưng `none` không khuyến nghị nếu GPU hạn chế

### 2. Dữ liệu
- SFT source: `vnu-llm2023-ftdata/qa-daotao-sft`
- KTO source: `vnu-llm2023-ftdata/qa-daotao-cho-rl`
- Dữ liệu được chuẩn hóa về:
  - `question`
  - `answer`
  - `context`
  - `insufficient_context`
  - `multi_intent`

### 3. Split cố định
- `data/splits/sft_train`
- `data/splits/sft_val`
- `data/splits/kto_train`
- `data/splits/kto_val`
- `data/splits/kto_test`
- `data/splits/test_only`
- `data/splits/split_manifest.json`

Quy ước sử dụng:
- SFT train dùng `sft_train`, eval dùng `sft_val`
- KTO train dùng `kto_train`, eval dùng `kto_val`
- Đánh giá SFT final thường dùng `test_only`
- Đánh giá KTO final thường dùng `kto_test`

### 4. Prompt format dùng chung
- Prompt được build theo format:
  - `### Chỉ dẫn hệ thống`
  - `### Câu hỏi`
  - `### Ngữ cảnh tham chiếu`
  - `### Trả lời`
- `system_prompt` được cấu hình trong:
  - `configs/sft_config.yaml`
  - `configs/kto_config.yaml`
  - `configs/eval_config.yaml`

### 5. Chuyển đổi dữ liệu cho KTO
- Không giả định sẵn preferred/dispreferred pairs
- Positive sample:
  - `completion = answer`
  - `label = True`
- Negative sample được sinh heuristic theo thứ tự:
  1. `partial_answer_for_multi_intent`
  2. `raw_context_dump`
  3. `overconfident_placeholder`
  4. `generic_non_answer`

### 6. Evaluation workflow hiện tại
- Flow cũ `run_eval.py` đã bị bỏ
- Flow hiện tại tách làm 2 bước:
  - `src.cli.run_infer`: chỉ generate response
  - `src.cli.run_judge`: chỉ chấm bằng Gemini
- Helper dùng chung nằm ở `src/utils/eval_utils.py`
- Inference dùng `model.generate()` trực tiếp, không dùng `transformers.pipeline`
- `scripts/workflow.sh` hỗ trợ:
  - `infer-model`
  - `judge-file`
  - `eval-sft`
  - `eval-kto`
  - `eval-all`

### 7. Gemini judge
- Gemini model hiện dùng: `gemini-3.1-flash-lite-preview`
- SDK hiện dùng: `google-genai`
- Prompt judge đọc từ `prompt/prompts.py`
- Quy trình judge:
  - classify `(question, context)` thành `CLASS_1`, `CLASS_2`, `CLASS_3`
  - chọn prompt judge tương ứng

## Các file quan trọng
- `configs/sft_config.yaml`
- `configs/kto_config.yaml`
- `configs/eval_config.yaml`
- `src/cli/run_sft.py`
- `src/cli/run_kto.py`
- `src/cli/run_infer.py`
- `src/cli/run_judge.py`
- `src/utils/data_utils.py`
- `src/utils/eval_utils.py`
- `src/utils/model_utils.py`
- `scripts/workflow.sh`

## Lệnh chuẩn nên dùng
- Chuẩn bị dữ liệu:
  - `bash scripts/workflow.sh prepare-data`
- Train SFT:
  - `bash scripts/workflow.sh train-sft`
- Train KTO:
  - `bash scripts/workflow.sh train-kto`
- Inference:
  - `bash scripts/workflow.sh infer-model <model_path> [results_dir] [split_name] [num_samples]`
- Judge:
  - `bash scripts/workflow.sh judge-file <generated_responses.json> [results_dir]`

## Các lưu ý thực tế
- `configs/sft_config.yaml` phải dùng `learning_rate: 0.0002`, không dùng `2e-4`
- KTO trên full model rất dễ OOM; ưu tiên `qlora`
- Evaluation end-to-end chậm vì inference và Gemini judge đều chạy tuần tự
- Nếu cần chỉ kiểm tra model output, dùng `infer-model` mà không chạy `judge-file`
