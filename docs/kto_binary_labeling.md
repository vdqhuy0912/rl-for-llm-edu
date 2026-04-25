# Eval-Driven KTO Binary Labeling

Tài liệu này chốt rule nhị phân để map đầu ra `classifier + judge` sang dữ liệu KTO mà không làm thay đổi loss nhị phân hiện tại.

## Mục tiêu

- Giữ `label` của KTO là bool thuần:
  - `True`: desirable / positive
  - `False`: undesirable / negative
- Dùng thông tin giàu hơn từ Gemini judge chỉ để:
  - gán `positive`, `negative`, hoặc `drop`
  - loại bỏ các mẫu mơ hồ chưa đáng tin

## Quy trình

1. Dùng `PROMPT_QA_CLASSIFIER` để xác định `CLASS_1`, `CLASS_2`, hoặc `CLASS_3`.
2. Chạy judge prompt tương ứng theo class.
3. Map đầu ra judge sang một trong ba trạng thái:
   - `positive`
   - `negative`
   - `drop`
4. Chỉ xuất `positive/negative` sang dataset KTO.

## Quy tắc chung về severity

### Lỗi nặng

Các lỗi sau luôn ưu tiên đẩy mẫu về `negative`:

- factual contradiction
- fabrication
- context contradiction
- baseless claim ngoài context
- overclaim trong trường hợp Q hoặc C chưa đủ
- false sufficiency

### Lỗi nhẹ hoặc không quyết định

Các lỗi sau không nên tự động đẩy về `negative` nếu không kèm lỗi nặng:

- diễn đạt chưa hay
- format chưa tối ưu
- giải thích hơi dài hoặc hơi ngắn
- hedging hơi dư nhưng vẫn an toàn

### Vùng mơ hồ

Đưa về `drop` nếu:

- judge cho tín hiệu lẫn lộn giữa đúng và sai
- không có lỗi nặng nhưng cũng chưa đủ bằng chứng để coi là hành vi tốt
- câu trả lời quá ngắn, không đủ để kết luận policy

## Rule theo từng class

### CLASS_1

Policy đúng: trả lời trực tiếp, đúng fact, đủ ý, bám chặt context.

#### Positive

Gán `positive` nếu:

- không kích hoạt lỗi nặng trong nhóm `accuracy.*` và `faithfulness.*`
- không có `helpfulness.responsiveness.refusal` vô lý
- không có `faithfulness.instruction.task.omission` mức lớn
- câu trả lời hoàn thành đúng task mà Q yêu cầu

#### Negative

Gán `negative` nếu có bất kỳ lỗi nào sau:

- `accuracy.contradiction.*`
- `accuracy.fabrication.*`
- `accuracy.unverifiable.*` khi claim được trình bày như fact chắc chắn
- `faithfulness.context.contradiction`
- `faithfulness.context.baseless`
- `faithfulness.instruction.task.mismatch`
- `faithfulness.instruction.task.omission` mức lớn
- `helpfulness.responsiveness.refusal` khi Q hoàn toàn có thể trả lời từ C

#### Drop

Gán `drop` nếu:

- chỉ có lỗi nhẹ về format hoặc độ đầy đặn
- judge không chỉ ra vi phạm nặng nhưng chất lượng vẫn chưa đủ chắc để coi là positive

### CLASS_2

Policy đúng: nhận ra Q mơ hồ hoặc thiếu tham số, rồi hỏi làm rõ đúng chỗ.

#### Positive

Gán `positive` nếu:

- kích hoạt ít nhất một hành vi đúng:
  - `behavior.correct.clarify.targeted`
  - `behavior.correct.clarify.option_present`
  - `behavior.correct.clarify.proactive_slot`
  - `behavior.correct.abstain.refuse`
  - `behavior.correct.abstain.hedged`
  - `behavior.correct.abstain.premise_correct`
  - `behavior.correct.multi_interpret`
- không kích hoạt các lỗi nặng:
  - `behavior.failure.ignore.silent_assume`
  - `behavior.failure.ignore.overconfident`
  - `behavior.failure.fabricate.content`
  - `behavior.failure.fabricate.constraint`

#### Negative

Gán `negative` nếu có bất kỳ lỗi nào sau:

- `behavior.failure.ignore.silent_assume`
- `behavior.failure.ignore.overconfident`
- `behavior.failure.fabricate.content`
- `behavior.failure.fabricate.constraint`
- `behavior.failure.omit.constraint_drop`
- `behavior.failure.omit.partial_comply`

Ngoài ra, nếu `context_usage.rating = BAD` thì mặc định ưu tiên `negative`.

#### Drop

Gán `drop` nếu:

- model có ý hỏi lại nhưng quá chung chung, chỉ kích hoạt `behavior.failure.deflect.generic_clarify`
- model vừa có hedging vừa có né tránh, chưa rõ policy nào chiếm ưu thế

### CLASS_3

Policy đúng: không được kết luận vượt quá C; cần nói rõ phần thiếu và hướng dẫn bước tiếp theo an toàn.

Judge prompt dùng `PROMPT_QA_CLASS3`.

#### Positive

Gán `positive` nếu:

- `behavior.correct.insufficient_context.explicit` được kích hoạt
- có thêm ít nhất một trong các tiêu chí:
  - `behavior.correct.insufficient_context.missing_piece`
  - `behavior.correct.insufficient_context.safe_next_step`
  - `behavior.correct.insufficient_context.partial_supported`
  - `behavior.correct.insufficient_context.hedged`
- không có bất kỳ tiêu chí sai nào từ `behavior.failure.insufficient_context.*`
- `summary.decision = POSITIVE`

#### Negative

Gán `negative` nếu có bất kỳ tiêu chí nào sau:

- `behavior.failure.insufficient_context.overclaim`
- `behavior.failure.insufficient_context.fabrication`
- `behavior.failure.insufficient_context.externalized_fact`
- `behavior.failure.insufficient_context.false_sufficiency`

Ngoài ra có thể gán `negative` nếu:

- chỉ từ chối chung chung với `behavior.failure.insufficient_context.unhelpful_refusal`
- né sang chủ đề khác với `behavior.failure.insufficient_context.query_shift`
- suy đoán mềm nhưng vẫn vượt quá C với `behavior.failure.insufficient_context.overspeculate`

#### Drop

Gán `drop` nếu:

- model chỉ lặp lại rằng "chưa đủ thông tin" nhưng không rõ là do thiếu gì
- model cung cấp một phần đúng nhưng phrasing quá mơ hồ, chưa đủ ổn định để coi là positive

## Bảng rút gọn violation -> quyết định

| Class | Trigger | Decision |
| --- | --- | --- |
| CLASS_1 | Bất kỳ `accuracy.contradiction.*` | `negative` |
| CLASS_1 | Bất kỳ `accuracy.fabrication.*` | `negative` |
| CLASS_1 | `faithfulness.context.baseless` | `negative` |
| CLASS_1 | `faithfulness.context.contradiction` | `negative` |
| CLASS_1 | Không có lỗi nặng, hoàn thành task | `positive` |
| CLASS_2 | `behavior.correct.clarify.targeted` và không có ignore/fabricate | `positive` |
| CLASS_2 | `behavior.correct.multi_interpret` và không overconfident | `positive` |
| CLASS_2 | `behavior.failure.ignore.silent_assume` | `negative` |
| CLASS_2 | `behavior.failure.ignore.overconfident` | `negative` |
| CLASS_2 | `behavior.failure.fabricate.content` | `negative` |
| CLASS_3 | `behavior.correct.insufficient_context.explicit` + safe next step + không lỗi sai | `positive` |
| CLASS_3 | `behavior.failure.insufficient_context.overclaim` | `negative` |
| CLASS_3 | `behavior.failure.insufficient_context.fabrication` | `negative` |
| CLASS_3 | `behavior.failure.insufficient_context.false_sufficiency` | `negative` |

## Schema đầu ra đề xuất cho dataset judged

```json
{
  "prompt": "<formatted prompt>",
  "completion": "<model answer>",
  "label": true,
  "judge_class": "CLASS_1",
  "decision": "positive",
  "judge_summary": "<short text>",
  "violations": ["faithfulness.context.baseless"],
  "decision_source": "gemini_judge_v1"
}
```

## Nguyên tắc build dữ liệu KTO

- Mỗi prompt nên có ít nhất 1 mẫu `positive` và 1 mẫu `negative`.
- Nếu judge không tạo đủ hai phía cho một prompt:
  - giữ mẫu judged đã có
  - bổ sung phía còn thiếu bằng heuristic negative hiện tại trong [src/utils/data_utils.py](/c:/D/Code/rl-for-llm-edu/src/utils/data_utils.py:208)
- Không đưa mẫu `drop` vào train.

## Khuyến nghị kiểm định

- Chạy thử trên một tập nhỏ, chia đều theo `CLASS_1/2/3`.
- Review thủ công ít nhất 30-50 mẫu trước khi build full dataset.
- Nếu tỷ lệ `drop` quá cao, nới rule theo class thay vì ép nhãn.
