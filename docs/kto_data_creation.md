# Cách tạo dữ liệu cho KTO

Tài liệu này mô tả cách pipeline hiện tại tạo dữ liệu KTO cho dự án `RL for LLM Education`.

Điểm quan trọng: `prepare-data` chưa tạo trực tiếp các dòng `prompt/completion/label` cho KTO. Lệnh này chỉ tạo các split cố định từ dataset RL. Dữ liệu KTO đúng schema của `TRL KTOTrainer` được tạo khi chạy `src.cli.run_kto`, thông qua hàm `prepare_kto_data()` trong `src/utils/data_utils.py`.

## Tổng quan luồng dữ liệu

Pipeline tạo data KTO có hai tầng:

1. Chuẩn bị split nguồn:
   - Input: `vnu-llm2023-ftdata/qa-daotao-cho-rl`
   - Output:
     - `data/splits/kto_train`
     - `data/splits/kto_val`
     - `data/splits/kto_test`
     - `data/splits/split_manifest.json`

2. Chuyển QA thành KTO rows:
   - Input: từng dòng QA trong `kto_train` hoặc `kto_val`
   - Output runtime cho trainer:
     - `prompt`
     - `completion`
     - `label`
     - metadata kiểm tra nguồn và chiến lược convert

Với mỗi dòng QA hợp lệ, pipeline tạo 2 dòng KTO:

| Row | `completion` | `label` | `conversion_strategy` |
| --- | --- | --- | --- |
| Positive | `answer` gốc | `True` | `gold_answer` |
| Negative | câu trả lời xấu sinh bằng heuristic | `False` | một trong các strategy negative |

Vì vậy nếu split nguồn có `N` dòng hợp lệ, dataset KTO sau convert thường có khoảng `2N` dòng. Các dòng thiếu `question` hoặc `answer` sẽ bị bỏ qua.

## Dataset nguồn

Dataset nguồn cho KTO được cấu hình trong `configs/kto_config.yaml`:

```yaml
data:
  train_dataset: "vnu-llm2023-ftdata/qa-daotao-cho-rl"
  val_dataset: "vnu-llm2023-ftdata/qa-daotao-cho-rl"
```

Theo quy ước split hiện tại:

| Mục đích | Saved split | Source split |
| --- | --- | --- |
| KTO train | `data/splits/kto_train` | `train` |
| KTO eval | `data/splits/kto_val` | `validation` |
| KTO final evaluation | `data/splits/kto_test` | `test` |

## Schema nguồn sau chuẩn hóa

Mỗi sample nguồn được chuẩn hóa bằng `normalize_qa_example()` về các field:

| Field chuẩn | Các field nguồn có thể đọc | Ý nghĩa |
| --- | --- | --- |
| `question` | `question`, `input`, `prompt`, `query` | Câu hỏi của người dùng |
| `answer` | `answer`, `output`, `response`, `completion` | Câu trả lời đúng/gold |
| `context` | `reference`, `references`, `context` | Ngữ cảnh tham chiếu |
| `insufficient_context` | `insufficient_context`, `insufficial context` | Cờ cho biết context thiếu |
| `multi_intent` | `multi_intent` | Cờ cho biết câu hỏi có nhiều ý định |

Sau chuẩn hóa, sample tối thiểu phải có:

```json
{
  "question": "...",
  "answer": "...",
  "context": "...",
  "insufficient_context": false,
  "multi_intent": false
}
```

`context` có thể rỗng, nhưng `question` và `answer` không được rỗng nếu muốn sample được đưa vào KTO.

## Bước 1: tải dữ liệu thô

Nếu chưa có dữ liệu local trong `data/raw`, chạy:

```bash
bash scripts/workflow.sh download-data
```

Lệnh này tải các dataset Hugging Face được dự án dùng và lưu vào `data/raw`.

## Bước 2: tạo split cố định

Chạy:

```bash
bash scripts/workflow.sh prepare-data
```

Lệnh này gọi `src.cli.prepare_data` và thực hiện:

1. Đọc dataset RL `vnu-llm2023-ftdata/qa-daotao-cho-rl`.
2. Kiểm tra đủ các split `train`, `validation`, `test`.
3. Gắn thêm field:

```json
{
  "source_dataset": "vnu-llm2023-ftdata/qa-daotao-cho-rl"
}
```

4. Lưu split ra disk:

```text
data/splits/kto_train
data/splits/kto_val
data/splits/kto_test
```

5. Ghi manifest:

```text
data/splits/split_manifest.json
```

Manifest cho biết split nào lấy từ dataset nào, ví dụ:

```json
{
  "rules": {
    "rl_dataset": "vnu-llm2023-ftdata/qa-daotao-cho-rl",
    "kto_train_source_split": "train",
    "kto_val_source_split": "validation",
    "kto_test_source_split": "test"
  }
}
```

## Bước 3: convert QA thành KTO rows

Khi chạy train:

```bash
bash scripts/workflow.sh train-kto
```

`src.cli.run_kto` sẽ:

1. Ưu tiên load `data/splits/kto_train`.
2. Nếu split local không tồn tại, fallback sang dataset trong `configs/kto_config.yaml`.
3. Load `data/splits/kto_val` cho eval.
4. Gọi `prepare_kto_data()` để convert từng QA sample thành format KTO.

Schema sau convert:

```json
{
  "prompt": "<formatted prompt>",
  "completion": "<assistant completion>",
  "label": true,
  "source_question": "<question gốc>",
  "source_context": "<context gốc>",
  "conversion_strategy": "gold_answer",
  "insufficient_context": false,
  "multi_intent": false
}
```

Ý nghĩa các field:

| Field | Ý nghĩa |
| --- | --- |
| `prompt` | Prompt đầy đủ đã format theo chat template hoặc fallback ChatML |
| `completion` | Phần trả lời assistant, không bao gồm prompt |
| `label` | `True` là desirable, `False` là undesirable |
| `source_question` | Câu hỏi gốc để debug |
| `source_context` | Context gốc để debug |
| `conversion_strategy` | Cách tạo completion |
| `insufficient_context` | Metadata giữ lại từ sample nguồn |
| `multi_intent` | Metadata giữ lại từ sample nguồn |

## Cách build prompt

Prompt dùng chung với SFT và evaluation, được build từ:

- `system_prompt` trong `configs/kto_config.yaml`
- `question`
- `context`

Nếu tokenizer có `chat_template`, pipeline dùng chat template của tokenizer. Nếu không, pipeline fallback về format ChatML nội bộ.

Nội dung user turn có dạng:

```text
Câu hỏi:
<question>

Ngữ cảnh tham chiếu:
<context>
```

Với `add_generation_prompt=True`, prompt được kết thúc tại vị trí assistant cần sinh câu trả lời. `completion` được tách riêng để `KTOTrainer` nhận đúng schema `prompt + completion + label`.

## Cách tạo positive sample

Positive sample luôn lấy từ `answer` gốc:

```json
{
  "prompt": "<prompt>",
  "completion": "<answer>",
  "label": true,
  "conversion_strategy": "gold_answer"
}
```

Mục tiêu của positive sample là dạy model ưu tiên câu trả lời đúng, bám context và tuân thủ system prompt.

## Cách tạo negative sample

Do dataset hiện tại không giả định có sẵn preferred/dispreferred pairs, negative sample được sinh bằng heuristic. Thứ tự ưu tiên trong code và config là:

1. `partial_answer_for_multi_intent`
2. `raw_context_dump`
3. `overconfident_placeholder`
4. `generic_non_answer`

### 1. `partial_answer_for_multi_intent`

Điều kiện:

- `multi_intent = true`
- `answer` có thể tách được câu đầu tiên
- câu đầu tiên khác toàn bộ `answer`

Cách tạo:

- Chỉ giữ câu đầu tiên của `answer`.

Ý nghĩa:

- Với câu hỏi nhiều ý, câu trả lời chỉ xử lý một phần là undesirable vì bỏ sót intent.

Ví dụ:

```json
{
  "answer": "Thí sinh cần chuẩn bị hồ sơ A. Sau đó nộp theo thời hạn B.",
  "negative_completion": "Thí sinh cần chuẩn bị hồ sơ A.",
  "label": false,
  "conversion_strategy": "partial_answer_for_multi_intent"
}
```

### 2. `raw_context_dump`

Điều kiện:

- Có `context`
- Đoạn context rút gọn khác với `answer`

Cách tạo:

- Lấy tối đa 400 ký tự đầu của `context`.
- Dùng đoạn này làm `completion`.

Ý nghĩa:

- Dump context thô không phải là câu trả lời trực tiếp. Model cần tổng hợp và trả lời câu hỏi, không chỉ copy ngữ cảnh.

Ví dụ:

```json
{
  "completion": "<400 ký tự đầu của context>",
  "label": false,
  "conversion_strategy": "raw_context_dump"
}
```

### 3. `overconfident_placeholder`

Điều kiện:

- `insufficient_context = true`
- Không tạo được negative bằng hai strategy phía trên

Cách tạo:

```text
Thong tin hien tai da du de ket luan ngay. Ban cu lam theo cach pho bien va khong can doi them xac nhan tu nha truong.
```

Ý nghĩa:

- Với context thiếu, hành vi xấu là kết luận quá chắc hoặc bảo người dùng làm theo khi chưa đủ căn cứ.

### 4. `generic_non_answer`

Điều kiện:

- Fallback khi không strategy nào phía trên dùng được

Cách tạo:

```text
Ban nen tham khao them thong bao cua truong vi minh chua the tra loi cu the ngay luc nay.
```

Ý nghĩa:

- Đây là negative an toàn để luôn có một phía undesirable cho sample, nhưng chất lượng tín hiệu thấp hơn các strategy cụ thể.

## Ví dụ một QA thành hai KTO rows

Input nguồn:

```json
{
  "question": "Hồ sơ xét tuyển cần những gì?",
  "answer": "Hồ sơ gồm phiếu đăng ký, học bạ và giấy tờ ưu tiên nếu có.",
  "context": "Hồ sơ xét tuyển bao gồm phiếu đăng ký xét tuyển, bản sao học bạ THPT và giấy tờ ưu tiên nếu có.",
  "insufficient_context": false,
  "multi_intent": false
}
```

Output KTO:

```json
[
  {
    "prompt": "<system + user question + context>",
    "completion": "Hồ sơ gồm phiếu đăng ký, học bạ và giấy tờ ưu tiên nếu có.",
    "label": true,
    "conversion_strategy": "gold_answer"
  },
  {
    "prompt": "<system + user question + context>",
    "completion": "Hồ sơ xét tuyển bao gồm phiếu đăng ký xét tuyển, bản sao học bạ THPT và giấy tờ ưu tiên nếu có.",
    "label": false,
    "conversion_strategy": "raw_context_dump"
  }
]
```

## Kiểm tra nhanh dữ liệu KTO

Dùng script preview:

```bash
python -m src.cli.preview_kto_data
```

Script này in:

- tên dataset nguồn
- số dòng nguồn
- số dòng KTO sau convert
- một vài sample đầu với:
  - `label`
  - `conversion_strategy`
  - `prompt`
  - `completion`
  - `insufficient_context`
  - `multi_intent`

Lưu ý: `preview_kto_data.py` gọi `prepare_kto_data()` không truyền tokenizer, nên prompt có thể dùng fallback ChatML thay vì đúng chat template của tokenizer. Khi train thật, `run_kto.py` truyền tokenizer từ model vào `prepare_kto_data()`.

## Train thử trên tập nhỏ

Để kiểm tra data conversion và trainer trước khi chạy full:

```bash
python -m src.cli.run_kto \
  --num-train-samples 16 \
  --num-eval-samples 8 \
  --max-steps 2
```

Do mỗi QA tạo 2 dòng KTO, `--num-train-samples 16` thường tạo khoảng 32 KTO rows cho train.

## Checklist chất lượng trước khi train full

Nên kiểm tra thủ công một số sample từ `preview_kto_data.py`:

- `prompt` có đủ system prompt, câu hỏi và context.
- `completion` positive không bị lặp lại prompt.
- `completion` negative thật sự là hành vi kém hơn positive.
- `conversion_strategy` phân bố hợp lý, không bị fallback `generic_non_answer` quá nhiều.
- Các sample `insufficient_context=true` không bị positive trả lời vượt quá context.
- Các sample `multi_intent=true` có positive trả lời đủ nhiều ý.

## Khi nào cần chỉnh heuristic negative

Nên chỉnh `_build_undesirable_completion()` nếu thấy:

- `raw_context_dump` quá giống `answer`, làm nhãn negative nhiễu.
- `generic_non_answer` xuất hiện quá nhiều.
- Dữ liệu có nhiều câu hỏi thiếu context nhưng `insufficient_context` không được gắn đúng.
- Câu hỏi multi-intent không có dấu câu rõ, khiến `partial_answer_for_multi_intent` không hoạt động.

## Pipeline judged-generated KTO

Ngoài luồng synthetic negative ở trên, code hiện hỗ trợ pipeline:

```text
kto_train/kto_val
  -> infer bằng SFT model
  -> Gemini judge generated_answer
  -> convert judge result thành prompt/completion/label
  -> train KTO từ generated_answer đã được judge
```

Pipeline này không dùng `CLASS_3` để train. Mọi record có `judge_classification_label = CLASS_3` bị loại hoàn toàn, kể cả khi có `reference_answer`.

### Bước 1: generate trên train và val

```bash
bash scripts/workflow.sh infer-model models/sft_checkpoints/final results/judged_kto_train kto_train
bash scripts/workflow.sh infer-model models/sft_checkpoints/final results/judged_kto_val kto_val
```

Output:

```text
results/judged_kto_train/generated_responses.json
results/judged_kto_val/generated_responses.json
```

Mỗi record có:

- `question`
- `context`
- `reference_answer`
- `generated_answer`
- `insufficient_context`
- `multi_intent`

### Bước 2: Gemini judge

```bash
bash scripts/workflow.sh judge-file results/judged_kto_train/generated_responses.json results/judged_kto_train
bash scripts/workflow.sh judge-file results/judged_kto_val/generated_responses.json results/judged_kto_val
```

Output:

```text
results/judged_kto_train/evaluation_results.json
results/judged_kto_val/evaluation_results.json
```

Mỗi record được bổ sung:

- `judge_classification`
- `judge_classification_label`
- `judge_evaluation_mode`
- `judge_evaluation`

### Bước 3: convert judge result thành KTO dataset

```bash
bash scripts/workflow.sh build-judged-kto-data \
  results/judged_kto_train/evaluation_results.json \
  results/judged_kto_val/evaluation_results.json
```

Output mặc định:

```text
data/splits/kto_judged_train
data/splits/kto_judged_val
```

Mỗi dòng KTO được tạo từ `generated_answer`, không phải synthetic negative:

```json
{
  "prompt": "<system + question + context>",
  "completion": "<generated_answer>",
  "label": true,
  "source_question": "<question>",
  "source_context": "<context>",
  "reference_answer": "<reference_answer>",
  "judge_classification_label": "CLASS_1",
  "judge_decision": "positive",
  "judge_decision_reason": "...",
  "judge_triggered_labels": [],
  "conversion_strategy": "gemini_judged_generated_answer"
}
```

### Rule convert label

`CLASS_3`:

- luôn bỏ qua
- không tạo `label=True`
- không tạo `label=False`
- không thêm `reference_answer` vào train

`CLASS_1`:

- `label=False` nếu có lỗi nặng:
  - `accuracy.contradiction.*`
  - `accuracy.fabrication.*`
  - `accuracy.unverifiable.*`
  - `faithfulness.context.contradiction`
  - `faithfulness.context.baseless`
  - `faithfulness.instruction.task.mismatch`
  - `helpfulness.responsiveness.refusal`
  - `safety.harm.*`
  - `safety.ethics.*`
- `label=True` nếu không có violated labels
- `drop` nếu chỉ có lỗi chưa đủ nặng, ví dụ omission không rõ mức độ

`CLASS_2`:

- `label=False` nếu có hành vi sai:
  - `behavior.failure.ignore.silent_assume`
  - `behavior.failure.ignore.overconfident`
  - `behavior.failure.fabricate.content`
  - `behavior.failure.fabricate.constraint`
  - `behavior.failure.omit.constraint_drop`
  - `behavior.failure.omit.partial_comply`
  - hoặc `context_usage.rating = BAD`
- `label=True` nếu có hành vi đúng:
  - `behavior.correct.clarify.targeted`
  - `behavior.correct.clarify.option_present`
  - `behavior.correct.clarify.proactive_slot`
  - `behavior.correct.abstain.refuse`
  - `behavior.correct.abstain.hedged`
  - `behavior.correct.abstain.premise_correct`
  - `behavior.correct.multi_interpret`
- `drop` nếu judge không đủ rõ để tạo nhãn binary

Sau khi build, xem thống kê ở:

```text
data/splits/kto_judged_train/build_stats.json
data/splits/kto_judged_val/build_stats.json
```

### Bước 4: train KTO bằng judged dataset

```bash
bash scripts/workflow.sh train-kto-judged
```

Lệnh này gọi:

```bash
python -m src.cli.run_kto \
  --train-kto-data data/splits/kto_judged_train \
  --eval-kto-data data/splits/kto_judged_val
```

Khi truyền `--train-kto-data` và `--eval-kto-data`, `run_kto.py` sẽ load dataset có sẵn `prompt/completion/label` và bỏ qua `prepare_kto_data()`.

Nếu muốn bổ sung `reference_answer` làm positive cho các record `CLASS_1/CLASS_2`, chạy converter trực tiếp với:

```bash
python -m src.cli.build_judged_kto_data \
  --input-path results/judged_kto_train/evaluation_results.json \
  --output-dir data/splits/kto_judged_train \
  --include-reference-positive
```

Tùy chọn này vẫn không dùng `CLASS_3`.

Nếu muốn dùng data judged bởi Gemini để tạo nhãn KTO, xem thêm `docs/kto_binary_labeling.md` để biết nền tảng rule binary.

## Các file liên quan

- `configs/kto_config.yaml`: cấu hình dataset, prompt, KTO loss và tuning mode.
- `src/cli/prepare_data.py`: tạo `data/splits/kto_train`, `kto_val`, `kto_test`.
- `src/utils/data_utils.py`: chuẩn hóa QA, build prompt, convert sang KTO rows.
- `src/cli/run_kto.py`: load split, gọi `prepare_kto_data()`, chạy `KTOTrainer`.
- `src/cli/build_judged_kto_data.py`: convert Gemini judged generated responses thành KTO dataset đã có `prompt/completion/label`.
- `src/cli/preview_kto_data.py`: preview conversion.
- `docs/kto_binary_labeling.md`: hướng mở rộng dùng Gemini judge để tạo nhãn binary.
