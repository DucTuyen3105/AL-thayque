# Dàn ý thuyết trình: Phân loại nhóm ngôn ngữ của tên bằng LSTM

## 1. Đặt vấn đề

Đọc một họ như `Yamamoto` hay `Nguyễn`, con người có thể đoán nhóm ngôn ngữ qua cấu trúc âm và chữ. Bài tập đặt câu hỏi: LSTM có học được các dấu hiệu đó từ tên được gán nhãn không?

Đầu ra là nhóm tên theo dataset, không phải thông tin được xác nhận về người mang tên. Những tên dùng chung như `Lee` cho thấy bài toán không có một đáp án duy nhất trong mọi trường hợp.

## 2. Dữ liệu

Dùng bảng nguồn trong README. Giải thích vì sao cộng số dòng của năm nguồn không bằng số mẫu thực tế: tên trùng giữa nguồn, bản có/không dấu, nhiều cách phiên âm và nhãn xung đột.

Sau lọc có 40.888 tên thuộc 10 nhóm. Các nhóm không cân bằng: Nhật và Nga có nhiều tên hơn Việt và Pháp. Số mẫu Việt còn 458 dù đã thêm nguồn; 2.552 tên ghép từ từ điển Việt chỉ chứa 455 thành phần thô, nên không được coi là hàng nghìn thành phần độc lập.

Phân biệt nhãn từ điển/tutorial và nhãn suy ra từ Wikidata. Dữ liệu nhiều hơn có thể tăng độ phủ nhưng cũng mang lỗi nhãn.

## 3. Tiền xử lý

Ví dụ `Nguyễn` được model đọc là `nguyễn`; embedding ánh xạ mỗi ký tự thành một vector 32 chiều.

Khóa kiểm tra trùng lại là `nguyen`. Như vậy model vẫn nhận tín hiệu dấu nhưng `Nguyễn` không thể nằm trong train nếu `Nguyen` nằm trong test. Bản không dấu bổ sung vào train để demo vẫn hiểu khi gõ không dấu.

Padding làm các tên có độ dài giống nhau trong batch. Trạng thái LSTM lấy tại ký tự cuối thật, không lấy sau một chuỗi padding.

## 4. Vì sao dùng LSTM?

Tên là chuỗi có thứ tự. `a`, `n`, `g` xuất hiện riêng lẻ chưa đủ; các cụm và vị trí như `ng`, `-ov`, `-ski`, `-moto` có thể chứa tín hiệu.

LSTM cập nhật trạng thái ẩn khi đọc lần lượt từng ký tự. Ba cổng điều khiển phần thông tin bị quên, được thêm vào bộ nhớ và được đưa ra trạng thái ẩn. Embedding được học cùng mô hình, không phải mã ASCII và không phải vector được gán sẵn bằng tay.

```text
x_t → Embedding → LSTM(h_{t-1}, c_{t-1}) → h_t, c_t
h_last → Linear → Softmax → 10 điểm dự đoán
```

LSTM 128 đơn vị và embedding 32 chiều tạo một model nhỏ, chạy CPU. PyTorch được dùng để thực hiện đúng kiến trúc LSTM như ý tưởng Keras ban đầu.

## 5. Huấn luyện và đánh giá

Train/validation/test 70/15/15 theo nhóm tên; seed 42. Chọn epoch theo Macro-F1 validation. Test được dùng để đánh giá checkpoint cuối của mỗi lần chạy, không để chọn epoch hoặc temperature.

Accuracy = số dự đoán đúng / tổng mẫu. Macro-F1 tính F1 của từng nhóm rồi lấy trung bình, giúp thấy nhóm ít mẫu có hoạt động tốt không. Top-3 đúng nếu nhãn thật nằm trong ba ứng viên.

Trình chiếu số thực tế trong `docs/KET_QUA.md` và hình `artifacts/confusion_matrix.png`. Chỉ ra nhóm nào còn nhầm, không dùng ví dụ demo đúng để thay thế đánh giá test.

Các phiên bản đã được phát triển và đánh giá nhiều lần trên cách chia cố định. Kết quả thích hợp cho đồ án; để công bố benchmark cuối, cần thêm một bộ test ngoài chưa từng xem, nhãn được xác minh và đủ mẫu mỗi nhóm.

## 6. Demo

```bash
source .venv/bin/activate
python app.py
```

Nhập `Nguyễn`, `Yamamoto`, `Min-jun`, `Smith`, `Pierre`, rồi `Lee` để minh họa sự mơ hồ. Tiếp theo nhập `Nguyễn Đức Tuyên`; giải thích demo đang trung bình điểm từng thành phần, không phải model đã train trên họ tên đầy đủ.

Một điểm Softmax cao chỉ nghĩa là model ưu tiên một nhóm trong mười nhóm được phép chọn. Nó không bảo đảm tên thuộc đúng nhóm đó, đặc biệt với tên ngoài dataset hoặc ngoài mười nhóm.

## 7. Hạn chế và hướng mở rộng

Tên dùng chung, phiên âm làm mất tín hiệu, nhãn nguồn có thể sai và dữ liệu lệch nhóm. Model hiện chỉ hỗ trợ chữ Latin. Cần dataset có đủ tên Việt độc lập và test ngoài nguồn, cùng nhãn đa ngôn ngữ được xác minh.

Nếu chuyển sang họ tên đầy đủ, dùng một dataset cùng loại cho mọi nhóm, chẳng hạn NaNa, thay vì ghép hàng nghìn họ và tên để tạo số mẫu giả. Khi đánh giá, có thể giữ họ chưa xuất hiện trong train để đo mức tổng quát hóa khó hơn.

## Câu hỏi thường gặp

**Vì sao không đạt 100%?** Có tên tồn tại trong nhiều nhóm, tên rất ngắn không đủ dấu hiệu, và nhãn cũng có nhiễu.

**LSTM có luôn tốt hơn mô hình đơn giản?** Không. Dự án có baseline TF-IDF ký tự + Logistic Regression để kiểm tra lợi ích của LSTM trên chính cách chia này.

**Có đang tra từ điển để trả kết quả không?** Không. Xếp hạng do LSTM tính. Bảng tên nhiều nhãn chỉ dùng để bổ sung cảnh báo.

**Có dùng dấu để nhận diện tên Việt không?** Có, dấu là một đặc trưng hợp lệ của chuỗi ký tự. Báo cáo thêm kết quả khi bỏ toàn bộ dấu trên test để thấy mức phụ thuộc vào đặc trưng này.

**Tên nước ngoài không có trong 10 nhóm thì sao?** Model vẫn phân bố điểm trong 10 nhóm; ngưỡng “chưa đủ chắc” là quy tắc minh họa, chưa được xác nhận là bộ phát hiện tên ngoài phân bố.

**Tại sao không lấy nhiều triệu tên để tăng accuracy?** Số tên độc lập, nhãn đúng và cách chia hợp lý quan trọng hơn số dòng. Dữ liệu khác loại hoặc nhãn quốc gia không thể tự coi là nhãn ngôn ngữ.
