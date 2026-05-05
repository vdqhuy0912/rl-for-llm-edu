# Tổng Quan Bộ Dữ Liệu

## Mục tiêu

Tài liệu này mô tả ngắn gọn các bộ dữ liệu đang được dùng trong pipeline của dự án và vai trò của chúng trong từng giai đoạn:

- Supervised Fine-Tuning (SFT)
- Knowledge Transfer Optimization (KTO)
- Inference và Gemini judge

Tài liệu này không đi sâu vào huấn luyện mô hình. Mục tiêu chính là làm rõ dữ liệu nào đang được dùng, dữ liệu được chuẩn hóa ra sao, và dữ liệu đi qua pipeline theo cách nào.

## Nguồn

<!-- Để trống theo yêu cầu -->

## Phân chia dữ liệu

<!-- Để trống theo yêu cầu -->

## Các nhóm dữ liệu trong pipeline

### 1. Dữ liệu cho SFT

Nhóm dữ liệu này được dùng để huấn luyện mô hình theo kiểu supervised learning.

Đặc điểm:
- Mỗi mẫu thường chứa câu hỏi và câu trả lời mục tiêu
- Có thể kèm ngữ cảnh tham chiếu
- Được dùng để dạy model cách trả lời theo format và domain của bài toán

Vai trò trong pipeline:
- Là đầu vào cho bước `run_sft`
- Tạo ra SFT final model để tiếp tục dùng làm base cho KTO

### 2. Dữ liệu cho KTO

Nhóm dữ liệu này được dùng để tạo dữ liệu preference-style cho KTO.

Đặc điểm:
- Không yêu cầu phải có sẵn preferred/dispreferred pairs
- Từ dữ liệu QA gốc, pipeline sẽ tự chuyển đổi thành các mẫu:
  - `prompt`
  - `completion`
  - `label`

Vai trò trong pipeline:
- Là đầu vào cho bước `run_kto`
- Dùng để tinh chỉnh tiếp từ SFT final model

### 3. Dữ liệu cho inference và judge

Nhóm dữ liệu này được dùng để chạy sinh câu trả lời từ model và sau đó chấm bằng Gemini.

Đặc điểm:
- Dữ liệu đầu vào vẫn là câu hỏi và ngữ cảnh
- Model sẽ sinh `generated_answer`
- Gemini judge sẽ đọc câu hỏi, ngữ cảnh và câu trả lời sinh ra để đánh giá

Vai trò trong pipeline:
- Dùng trong `run_infer`
- Dùng tiếp trong `run_judge`

## Schema chuẩn hóa đang dùng

Các bộ dữ liệu thực tế có thể khác nhau nhẹ về tên cột. Trong code, dữ liệu được chuẩn hóa về cùng một schema logic:

- `question`
- `answer`
- `context`
- `insufficient_context`
- `multi_intent`

Ý nghĩa:
- `question`: câu hỏi đầu vào
- `answer`: câu trả lời chuẩn hoặc câu trả lời tham chiếu
- `context`: ngữ cảnh tham chiếu dùng để trả lời
- `insufficient_context`: cờ cho biết ngữ cảnh có thể chưa đủ để trả lời đầy đủ
- `multi_intent`: cờ cho biết câu hỏi có nhiều ý

## Kết quả khảo sát dữ liệu local hiện có

Qua kiểm tra trực tiếp các dataset local trong `data/raw` và các split đã lưu trong `data/splits`, schema thực tế hiện tại khá đồng nhất giữa bộ SFT và bộ RL/KTO.

Lần phân tích gần nhất được chạy trên toàn bộ dữ liệu bằng script:

- `python -m src.cli.analyze_data --output-dir results/data_analysis_full --sample-size 1000000`

Artifact phân tích:

- `results/data_analysis_full/data_analysis.json`
- `results/data_analysis_full/data_analysis.md`

### Schema thực tế của dataset gốc

Hai dataset gốc hiện có cùng tập cột:

- `id`
- `question`
- `answer`
- `reference`
- `multi_intent`
- `insufficient_context`
- `reasoning_level`
- `topic`
- `question_type`

Nhận xét:
- Cấu trúc giữa dataset SFT và dataset RL/KTO hiện tại là giống nhau
- Điều này giúp pipeline tái sử dụng cùng logic load, normalize và build prompt
- Trường `reference` đang là tên cột ngữ cảnh thực tế trên disk, sau đó mới được ánh xạ logic sang `context`

### Schema thực tế của các split đã lưu

Các split trong `data/splits` hiện có thêm một cột metadata:

- `source_dataset`

Vì vậy schema của split đã lưu là:

- `id`
- `question`
- `answer`
- `reference`
- `multi_intent`
- `insufficient_context`
- `reasoning_level`
- `topic`
- `question_type`
- `source_dataset`

Ý nghĩa:
- `source_dataset` giúp truy vết mỗi bản ghi đến từ dataset nào sau khi đã chuẩn bị split
- Điều này hữu ích khi kiểm tra pipeline, debug data hoặc muốn phân tích chéo theo nguồn dữ liệu

## Tóm tắt định lượng từ lần phân tích gần nhất

### 1. Quy mô dữ liệu

Theo dữ liệu local hiện có:

- Bộ SFT:
  - `train`: 11,431 mẫu
  - `validation`: 615 mẫu
  - `test`: 1,971 mẫu
- Bộ RL/KTO:
  - `train`: 2,000 mẫu
  - `validation`: 615 mẫu
  - `test`: 1,971 mẫu

Các split đã lưu trong `data/splits` đang phản ánh đúng quy mô này:

- `sft_train`: 11,431
- `sft_val`: 615
- `test_only`: 1,971
- `kto_train`: 2,000
- `kto_val`: 615
- `kto_test`: 1,971

### 2. Độ dài văn bản theo từng nhóm dữ liệu

Số liệu dưới đây được tính trên toàn bộ từng split.

| Split | Avg question chars | Avg answer chars | Avg reference chars |
| --- | ---: | ---: | ---: |
| `sft_train` | 160.84 | 327.88 | 3810.15 |
| `sft_val` | 157.69 | 324.69 | 3813.45 |
| `test_only` | 128.99 | 247.96 | 2306.64 |
| `kto_train` | 164.16 | 328.40 | 4081.45 |
| `kto_val` | 157.69 | 324.69 | 3813.45 |
| `kto_test` | 128.99 | 247.96 | 2306.64 |

| Nhận xét | Nội dung |
| --- | --- |
| Context dài | Các split train/val của SFT và KTO đều có `reference` rất dài, trung bình khoảng 3.8k đến 4.1k ký tự. |
| Dài nhất | `kto_train` có ngữ cảnh dài nhất trong các split đã lưu. |
| Test nhẹ hơn train | `test_only` và `kto_test` ngắn hơn đáng kể so với train/val, nhưng vẫn đủ dài để tạo áp lực lên inference. |
| Khác biệt chính | Độ dài `question` và `answer` giữa `sft_train` và `kto_train` tương đối gần nhau; khác biệt chính nằm ở độ dài context. |

### 3. Phân bố `multi_intent` và `insufficient_context`

| Split | `multi_intent = true` | `insufficient_context = true` |
| --- | ---: | ---: |
| `sft_train` | 731 / 11,431 | 1,567 / 11,431 |
| `sft_val` | 24 / 615 | 85 / 615 |
| `test_only` | 307 / 1,971 | 120 / 1,971 |
| `kto_train` | 310 / 2,000 | 216 / 2,000 |
| `kto_val` | 24 / 615 | 85 / 615 |
| `kto_test` | 307 / 1,971 | 120 / 1,971 |

| Nhận xét | Nội dung |
| --- | --- |
| `sft_train` | Phần lớn vẫn là câu hỏi đơn ý, nhưng số mẫu `insufficient_context` thực tế không nhỏ. |
| `kto_train` so với `sft_train` | Có tỷ lệ `multi_intent` cao hơn `sft_train`, nhưng không áp đảo như kết luận từ sample nhỏ. |
| Độ khó `kto_train` | Có số lượng mẫu `insufficient_context` đáng kể, nghĩa là tập này không chỉ khó vì nhiều ý mà còn khó vì thiếu ngữ cảnh. |
| `test_only` và `kto_test` | Có cấu trúc giống nhau về hai cờ này. |

### 4. Phân bố `reasoning_level`

| Split | Level 0 | Level 1 | Level 2 |
| --- | ---: | ---: | ---: |
| `sft_train` | 7,797 | 2,756 | 878 |
| `sft_val` | 434 | 145 | 36 |
| `test_only` | 1,273 | 413 | 285 |
| `kto_train` | 1,190 | 485 | 325 |
| `kto_val` | 434 | 145 | 36 |
| `kto_test` | 1,273 | 413 | 285 |

| Nhận xét | Nội dung |
| --- | --- |
| Xu hướng chung | Tất cả các split đều thiên về `reasoning_level = 0`. |
| Split khó hơn | `test_only` và `kto_train` có tỷ trọng mức `1` và `2` cao hơn rõ rệt so với `sft_train`. |
| Split dễ nhất | `sft_train` vẫn là tập dễ nhất về mặt suy luận. |
| `kto_train` | Khó hơn `sft_train`, nhưng không phải đa số tuyệt đối ở mức `2`; phân bố khó hơn và cân bằng hơn. |
| Cặp split giống nhau | `sft_val` và `kto_val` có cùng phân bố; `test_only` và `kto_test` cũng vậy. |

### 5. Chủ đề nổi bật

| Split | Topic 1 | Topic 2 | Topic 3 | Topic 4 | Topic 5 |
| --- | --- | --- | --- | --- | --- |
| `sft_train` | `Quy định chung & Chương trình đào tạo` (1,769) | `Tuyển sinh` (1,444) | `Quyền lợi, nghĩa vụ & chính sách người học` (1,385) | `Khác` (1,358) | `Nghiên cứu khoa học & Luận văn/Luận án` (1,168) |
| `sft_val` | `Quy định chung & Chương trình đào tạo` (93) | `Khác` (83) | `Tuyển sinh` (80) | `Quyền lợi, nghĩa vụ & chính sách người học` (79) | `Nghiên cứu khoa học & Luận văn/Luận án` (61) |
| `test_only` | `Tổ chức đào tạo & Quản lý học tập` (318) | `Quy định chung & Chương trình đào tạo` (256) | `Nghiên cứu khoa học & Luận văn/Luận án` (163) | `Tuyển sinh` (149) | `XTT & ƯTXT` (125) |
| `kto_train` | `Quy định chung & Chương trình đào tạo` (270) | `Quyền lợi, nghĩa vụ & chính sách người học` (213) | `Tuyển sinh` (213) | `Khác` (198) | `Nghiên cứu khoa học & Luận văn/Luận án` (176) |
| `kto_val` | `Quy định chung & Chương trình đào tạo` (93) | `Khác` (83) | `Tuyển sinh` (80) | `Quyền lợi, nghĩa vụ & chính sách người học` (79) | `Nghiên cứu khoa học & Luận văn/Luận án` (61) |
| `kto_test` | `Tổ chức đào tạo & Quản lý học tập` (318) | `Quy định chung & Chương trình đào tạo` (256) | `Nghiên cứu khoa học & Luận văn/Luận án` (163) | `Tuyển sinh` (149) | `XTT & ƯTXT` (125) |

| Nhận xét | Nội dung |
| --- | --- |
| `sft_train` và `kto_train` | Không chỉ tập trung vào tuyển sinh; các nhóm lớn còn bao gồm quy định đào tạo, quyền lợi người học và nghiên cứu khoa học. |
| `test_only` | Nghiêng nhiều hơn về các câu hỏi vận hành học tập, quy định đào tạo và tuyển sinh thực tế. |
| `kto_train` | Phân bố chủ đề đa dạng hơn nhận định ban đầu; không chỉ tập trung vào `XTT & ƯTXT`. |

### 6. Phân bố `question_type`

| Split | Type 1 | Type 2 | Type 3 | Type 4 | Type 5 |
| --- | --- | --- | --- | --- | --- |
| `sft_train` | `Cái gì` (3,730) | `Như thế nào` (1,937) | `Bao nhiêu` (1,398) | `Có/Không` (1,159) | `Tại sao` (1,012) |
| `sft_val` | `Cái gì` (206) | `Như thế nào` (98) | `Bao nhiêu` (89) | `Tại sao` (65) | `Có/Không` (65) |
| `test_only` | `Cái gì` (499) | `Như thế nào` (336) | `Bao nhiêu` (258) | `Có/Không` (169) | `Khi nào` (145) |
| `kto_train` | `Cái gì` (566) | `Như thế nào` (284) | `Bao nhiêu` (205) | `Có/Không` (191) | `Tại sao` (143) |
| `kto_val` | `Cái gì` (206) | `Như thế nào` (98) | `Bao nhiêu` (89) | `Tại sao` (65) | `Có/Không` (65) |
| `kto_test` | `Cái gì` (499) | `Như thế nào` (336) | `Bao nhiêu` (258) | `Có/Không` (169) | `Khi nào` (145) |

| Nhận xét | Nội dung |
| --- | --- |
| `kto_train` | `question_type` thực tế không bị trống; kết luận từ sample nhỏ trước đó là sai lệch. |
| Xu hướng chung | Ở tất cả các split lớn, `Cái gì` là loại câu hỏi phổ biến nhất. |
| Mức độ đa dạng | `Như thế nào`, `Bao nhiêu`, `Có/Không` và `Tại sao` cũng chiếm tỷ trọng lớn, cho thấy bài toán không chỉ là fact retrieval mà còn có nhiều câu hỏi thủ tục, điều kiện và giải thích. |

## Phân tích các cột thông tin

### 1. `id`

Vai trò:
- định danh duy nhất cho từng mẫu dữ liệu

Quan sát:
- `id` đang được đặt theo prefix của split, ví dụ:
  - `sft_train_00000`
  - `sft_val_00000`
  - `sft_test_00000`
  - `kto_train_00000`
  - `kto_val_00000`
  - `kto_test_00000`

Ý nghĩa:
- thuận tiện cho truy vết bản ghi
- dễ nối kết giữa output infer/judge và dữ liệu gốc nếu cần mở rộng pipeline sau này

### 2. `question`

Vai trò:
- chứa câu hỏi đầu vào từ người dùng hoặc câu hỏi mô phỏng

Đặc điểm:
- là trường trung tâm của toàn bộ pipeline
- được dùng ở SFT, KTO, inference và judge

Ý nghĩa nghiệp vụ:
- phản ánh nhu cầu hỏi đáp trong miền tuyển sinh, đào tạo, giới thiệu đơn vị, chương trình học và các thủ tục liên quan

### 3. `answer`

Vai trò:
- chứa câu trả lời mục tiêu hoặc câu trả lời tham chiếu

Đặc điểm:
- trong SFT, đây là target để model học sinh token tiếp theo
- trong KTO, đây là nguồn để tạo positive completion
- trong inference/judge, đây là `reference_answer` để đối chiếu với câu trả lời sinh ra

### 4. `reference`

Vai trò:
- chứa ngữ cảnh tham chiếu để trả lời

Đặc điểm:
- đây là trường context thực tế trong dữ liệu local hiện có
- có thể là đoạn văn bản dài, trích từ quy chế, đề án tuyển sinh, thông tin chương trình đào tạo hoặc mô tả đơn vị

Ý nghĩa:
- là nguồn thông tin để model bám sát dữ kiện
- cũng là cơ sở để judge kiểm tra câu trả lời có trung thành với tài liệu hay không

### 5. `multi_intent`

Vai trò:
- cho biết câu hỏi có nhiều ý cần trả lời hay không

Quan sát:
- giá trị đang xuất hiện dưới dạng chuỗi như `\"true\"` hoặc `\"false\"` trong dữ liệu local
- trong code, trường này được chuẩn hóa lại về boolean logic

Ý nghĩa trong pipeline:
- ảnh hưởng trực tiếp đến chiến lược tạo negative sample cho KTO
- nếu câu hỏi nhiều ý, pipeline có thể dùng chiến lược `partial_answer_for_multi_intent`

### 6. `insufficient_context`

Vai trò:
- cho biết ngữ cảnh tham chiếu có thể chưa đủ để trả lời trọn vẹn

Quan sát:
- tương tự `multi_intent`, trường này hiện xuất hiện dưới dạng chuỗi `\"true\"` hoặc `\"false\"` trong dữ liệu local
- code sẽ convert lại sang boolean khi normalize

Ý nghĩa trong pipeline:
- giúp xác định khi nào model nên trả lời thận trọng
- ảnh hưởng đến chiến lược negative sample kiểu `overconfident_placeholder` trong KTO

### 7. `reasoning_level`

Vai trò:
- biểu diễn mức độ suy luận cần thiết cho câu hỏi

Quan sát:
- là trường số nguyên
- ví dụ thực tế thấy có các mức như `0`, `1`, `2`

Diễn giải hợp lý:
- mức thấp có thể là câu hỏi truy xuất thông tin trực tiếp
- mức cao hơn có thể yêu cầu tổng hợp nhiều đoạn hoặc diễn giải phức tạp hơn

Lưu ý:
- hiện tại pipeline train/infer chưa dùng trường này trực tiếp để đổi chiến lược generate
- tuy nhiên đây là metadata hữu ích cho phân tích chất lượng sau này

### 8. `topic`

Vai trò:
- nhãn chủ đề của câu hỏi

Quan sát:
- dữ liệu mẫu hiện có các topic như:
  - `Giới thiệu`
  - `Chương trình học`
  - `XTT-ƯTXT`

Ý nghĩa:
- hỗ trợ phân tích phân bố câu hỏi theo chủ đề
- có thể dùng để đo chất lượng model theo từng nhóm nghiệp vụ

### 9. `question_type`

Vai trò:
- mô tả loại câu hỏi

Quan sát:
- trường này có mặt trong schema thực tế
- chưa được dùng trực tiếp trong code train/infer hiện tại

Ý nghĩa:
- là metadata hữu ích để phân loại câu hỏi
- có thể được khai thác trong tương lai cho đánh giá theo loại truy vấn

### 10. `source_dataset`

Vai trò:
- chỉ xuất hiện trong các split đã lưu, không phải trong raw dataset gốc

Ý nghĩa:
- dùng để ghi lại nguồn của mỗi mẫu sau khi build split cục bộ
- có ích khi một pipeline sau này cần trộn nhiều nguồn dữ liệu hoặc audit lại đầu vào

## Mapping các biến thể schema

Trong quá trình load dữ liệu, code hiện tại map các field tương đương về schema chung:

- `question`, `input`, `prompt`, `query` -> `question`
- `answer`, `output`, `response`, `completion` -> `answer`
- `reference`, `references`, `context` -> `context`
- `insufficient_context`, `insufficial context` -> `insufficient_context`

Việc chuẩn hóa này giúp cùng một pipeline có thể dùng lại cho nhiều nguồn dữ liệu mà không phải viết logic riêng cho từng bộ.

## Một số nhận xét chất lượng dữ liệu ở mức cấu trúc

### 1. Ưu điểm

- Hai dataset gốc đang có schema nhất quán
- Các trường chính cho QA có mặt đầy đủ: câu hỏi, câu trả lời, ngữ cảnh
- Có thêm metadata phục vụ phân tích như `topic`, `question_type`, `reasoning_level`
- Có cờ hỗ trợ KTO heuristic như `multi_intent` và `insufficient_context`

### 2. Điểm cần lưu ý

- Một số cờ logic đang được lưu dưới dạng chuỗi thay vì boolean thực
- Tên cột context thực tế là `reference`, trong khi code nội bộ thường dùng khái niệm `context`
- Metadata như `reasoning_level`, `topic`, `question_type` hiện chưa được khai thác đầy đủ trong train/infer

### 3. Ý nghĩa đối với pipeline hiện tại

- Dữ liệu hiện tại phù hợp để chạy SFT
- Dữ liệu hiện tại cũng phù hợp để sinh dữ liệu KTO từ QA gốc
- Pipeline evaluation có đủ trường để judge theo bộ ba:
  - `question`
  - `reference/context`
  - `generated_answer`

## Format prompt dùng chung

Sau khi chuẩn hóa, dữ liệu được đưa vào cùng một format prompt:

- `### Chỉ dẫn hệ thống`
- `### Câu hỏi`
- `### Ngữ cảnh tham chiếu`
- `### Trả lời`

Ý nghĩa:
- Giữ đầu vào nhất quán giữa train và inference
- Giảm lệch format giữa SFT, KTO và eval

## Dữ liệu được dùng trong SFT như thế nào

Trong SFT:
- Prompt được tạo từ `question` và `context`
- `answer` được nối vào cuối prompt để tạo target sequence

Nói ngắn gọn:
- model học từ `prompt + answer`

## Dữ liệu được dùng trong KTO như thế nào

Trong KTO:
- Dữ liệu QA gốc được chuyển thành các mẫu preference-style
- Mỗi mẫu QA hợp lệ thường sinh ra:
  - một mẫu positive
  - một mẫu negative

### Positive sample

- `completion = answer`
- `label = True`

### Negative sample

Negative sample được sinh heuristic theo thứ tự ưu tiên:

1. `partial_answer_for_multi_intent`
2. `raw_context_dump`
3. `overconfident_placeholder`
4. `generic_non_answer`

Mục tiêu:
- tạo tín hiệu để model phân biệt câu trả lời tốt và câu trả lời kém chất lượng
- không phụ thuộc vào dữ liệu preference pair được gán nhãn thủ công

## Dữ liệu được dùng trong inference như thế nào

Trong inference:
- Model chỉ nhận prompt
- Model tự sinh phần trả lời mới
- Kết quả được lưu thành:
  - `generated_responses.json`
  - `generated_responses.csv`

Mỗi record thường gồm:
- `question`
- `context`
- `reference_answer`
- `generated_answer`
- `insufficient_context`
- `multi_intent`

## Dữ liệu được dùng trong judge như thế nào

Gemini judge đọc output inference và đánh giá theo hai bước:

1. Phân loại cặp `(question, context)` vào một nhóm đánh giá
2. Dùng prompt judge tương ứng để đánh giá `generated_answer`

Kết quả được lưu thành:
- `evaluation_results.json`
- `evaluation_results.csv`

Mỗi record đánh giá thường bao gồm thêm:
- `judge_classification`
- `judge_classification_label`
- `judge_evaluation_mode`
- `judge_evaluation`

## Luồng dữ liệu tổng quát

### Luồng SFT

`question/context/answer` -> chuẩn hóa -> build prompt -> train SFT

### Luồng KTO

`question/context/answer` -> chuẩn hóa -> chuyển thành positive/negative completions -> train KTO

### Luồng inference và judge

`question/context` -> build prompt -> generate answer -> Gemini judge

## Một số lưu ý quan trọng

- KTO hiện tại dùng dữ liệu QA gốc đã được chuẩn hóa, không dùng trực tiếp output infer/eval của SFT làm train set
- SFT final model được dùng làm model khởi tạo cho KTO
- Inference và judge là hai bước tách riêng
- Cùng một schema dữ liệu được tái sử dụng cho nhiều bước để giảm sai lệch format
