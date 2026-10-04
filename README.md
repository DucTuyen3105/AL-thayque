# Dự đoán nhóm ngôn ngữ của tên người bằng LSTM

Bài tập chạy hoàn chỉnh: tải dữ liệu → làm sạch → chia train/validation/test → huấn luyện LSTM → đánh giá → demo web tiếng Anh.

Mô hình phân loại **đặc trưng nguồn gốc/ngôn ngữ của tên**, trong 10 nhóm Chinese, English, French, German, Italian, Japanese, Korean, Russian, Spanish, Vietnamese. Nhãn này không xác định quốc tịch, dân tộc hoặc ngôn ngữ một người sử dụng.

Đã huấn luyện trên bộ **40.888 tên sau lọc** từ năm nguồn. Bản cuối đạt **83,24% accuracy**, **67,79% Macro-F1** trên 6.138 tên test. Nhóm Việt còn hạn chế: F1 **48,89%** trên 68 mẫu. Xem [báo cáo kết quả](docs/KET_QUA.md) để biết điểm từng nhóm và phạm vi đánh giá.

## Chạy demo đã huấn luyện

Mở terminal trong thư mục dự án:

```bash
source .venv/bin/activate
python app.py
```

Mở **http://127.0.0.1:5000**. Nếu cổng đã được dùng: `python app.py --port 5001`.

Hoặc chạy `./run_demo.sh`. Ảnh giao diện đã kiểm tra trên Chrome: `artifacts/demo_desktop.png` và `artifacts/demo_mobile.png`.

Demo dùng mô hình LSTM đã lưu trong `artifacts/model.pt`, không trả nhãn bằng tra cứu từ điển. Tên có nhiều nhãn trong nguồn dữ liệu được cảnh báo riêng. Các phần trăm là điểm mô hình, không phải xác suất về thông tin cá nhân của một người.

CLI:

```bash
python -m name_origin.predict "Nguyễn" "Yamamoto" "John Smith" "Kim Min-jun"
```

**Phạm vi input:** model chính được train và test trên họ, tên riêng và thành phần tên bằng chữ Latin, có dấu hoặc không dấu. Với họ tên có dấu cách, chế độ `auto` lấy trung bình điểm các thành phần. Đây là một cách gộp phục vụ demo, **chưa được đánh giá trên dataset họ tên đầy đủ**. Không dùng điểm test của tên riêng để tuyên bố độ chính xác cho họ tên đầy đủ. Tên có gạch nối như `Min-jun` được đọc thành một chuỗi.

## Tái tạo toàn bộ bài tập

Trên máy mới, dùng Python 3.11 trở lên; môi trường đã chạy trong bài này là Python 3.14 và PyTorch bản CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m name_origin.prepare
python -m name_origin.train
python -m unittest discover -s tests -v
python app.py
```

Dữ liệu và mô hình đã có sẵn trong workspace. Chạy lại bước làm sạch khi không có mạng:

```bash
python -m name_origin.prepare --offline
```

Huấn luyện mặc định: 50 epoch tối đa, early stopping 8 epoch, batch 128, seed 42, 4 CPU thread. Checkpoint tốt nhất được chọn bằng **Macro-F1 validation**, không chọn bằng accuracy test.

`requirements-lock.txt` ghi phiên bản thực tế của môi trường chạy. Chỉ dùng lock khi nền tảng/Python tương thích. Không cần Playwright để chạy bài tập; nó chỉ dùng kiểm tra giao diện trong phiên phát triển.

Để mở notebook, cài thêm `python -m pip install notebook ipykernel`, chạy `python -m notebook` trong thư mục dự án và chọn `Name_Origin_LSTM.ipynb`.

## Dataset đã tìm và sử dụng

Xem [NGUON_DATASET.md](NGUON_DATASET.md) để biết đường tải từng file, cách lấy nhãn và số tên thực tế của mỗi nguồn sau xử lý.

| Nguồn | Dữ liệu tải được | Cách dùng | Hạn chế |
|---|---:|---|---|
| [PyTorch Names](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) | 18 danh sách họ; khoảng 20 nghìn dòng toàn bộ | Chọn 10 nhóm, đọc nhãn từ tên file | Rất mất cân bằng; nguồn thu thập gốc không được mô tả đầy đủ trong file zip |
| [Behind the Name / Anwarvic](https://github.com/Anwarvic/Behind-The-Name) | 20.505 tên riêng | Đọc trường `origin`; chỉ lấy nhãn ngôn ngữ hiện đại khớp 10 nhóm | Một tên có thể có nhiều nhãn; kho không nêu giấy phép dữ liệu riêng rõ ràng |
| [Popular Names by Country](https://github.com/sigpwned/popular-names-by-country-dataset) | 2.480 dòng tên riêng + 2.576 dòng họ trong bản CSV đã tải | Lấy 10 quốc gia đại diện và các bản Latin; nối các cách viết cùng `Name Group` | Nhãn quốc gia phổ biến là nhãn thay thế cho truyền thống tên, không phải từ nguyên; dữ liệu có biến thể |
| [Vietnamese NameDB / duyet](https://github.com/duyet/vietnamese-namedb) | 1.236 tên nam + 1.316 tên nữ ở hai file gốc | Tách thành phần và loại trùng; chỉ 455 thành phần thô, không đếm mọi tổ hợp như tên độc lập mới | Từ điển cộng đồng, không phải khảo sát dân số; một số thành phần không hợp lệ bị loại |
| [Wikidata / Stanford RegLab](https://github.com/reglab/disaggregation) | 25.876 tên riêng + 18.703 họ | Lấy 4 nhóm Á Đông phù hợp; chỉ nhận dòng có điểm liên hệ ≥ 0,95 trong bảng 6 nhóm | **Nhãn suy ra (silver label)** từ liên hệ tên–nhóm của cohort Wikidata, không phải nhãn ngôn ngữ do chuyên gia xác nhận |

Bộ RegLab đi kèm nghiên cứu [Lin et al., Scientific Data, 2025](https://doi.org/10.1038/s41597-025-04753-y). Nghiên cứu gốc tổng hợp thông tin từ hơn 300 nghìn cá nhân, nhưng **số người không phải số tên độc lập**. Dự án chỉ dùng hai bảng tên công khai, không dùng dữ liệu bệnh nhân hoặc địa lý.

Số dòng của bản tải có thể khác số tên mà README của nguồn quảng bá. Thống kê thực tế nằm trong `data/stats.json`; đường tải và SHA-256 của từng file nằm trong `data/raw/manifest.json`. Nếu checksum đổi, script yêu cầu kiểm tra nguồn trước khi chấp nhận bản mới.

Kết quả sau làm sạch: **40.888 tên, 40.848 nhóm độc lập**, loại **2.369 tên có nhãn xung đột**. Train/validation/test: **28.611 / 6.139 / 6.138** tên. Đây là số tên còn lại sau gộp, không phải tổng cộng số dòng của các nguồn.

| Nhóm | Tổng tên | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Chinese | 5.345 | 3.741 | 802 | 802 |
| English | 5.624 | 3.937 | 843 | 844 |
| French | 597 | 418 | 90 | 89 |
| German | 1.007 | 705 | 151 | 151 |
| Italian | 1.171 | 820 | 175 | 176 |
| Japanese | 12.587 | 8.806 | 1.891 | 1.890 |
| Korean | 3.804 | 2.657 | 574 | 573 |
| Russian | 9.608 | 6.725 | 1.441 | 1.442 |
| Spanish | 687 | 481 | 103 | 103 |
| Vietnamese | 458 | 321 | 69 | 68 |

### Nguồn lớn khác đã đối chiếu

- [NaNa / name2nat](https://github.com/Kyubyong/name2nat), [bản dataset trên Kaggle](https://www.kaggle.com/datasets/bryanpark/nana-dataset): tác giả báo cáo 1.112.902 tên từ Wikipedia, gồm 1.572 tên Việt và 9.871 tên Hàn. Phù hợp để làm **một bài riêng về họ tên đầy đủ với nhãn quốc gia suy ra**, không trộn trực tiếp vào benchmark tên riêng này. Chưa tải/train NaNa trong dự án.
- [UIT-ViNames](https://github.com/kietnv/VietnameseDatasets): dataset nghiên cứu họ tên Việt và giới tính; có thể hỗ trợ bản mở rộng họ tên nhưng không có nhãn nhiều ngôn ngữ. Chưa dùng train.
- [philipperemy/name-dataset](https://github.com/philipperemy/name-dataset): kho rất lớn, nhưng dữ liệu phân bố theo quốc gia và README nêu nguồn từ Facebook dump; không dùng trong bản này. Kích thước lớn không tự bảo đảm nhãn nguồn gốc chính xác.

## Mô hình và cách tránh kết quả cao giả

```text
Tên → chữ thường và chuẩn hóa dấu câu (giữ dấu Latin)
    → ánh xạ từng ký tự thành số, padding
    → Embedding(32)
    → LSTM(128)
    → trạng thái tại ký tự cuối thật
    → Dropout(0,3)
    → Linear(10)
    → Softmax có temperature calibration
```

- Khóa chống trùng bỏ dấu và chuyển chữ thường: `Nguyễn`, `NGUYỄN`, `Nguyen` thuộc cùng một tên. Model vẫn được đọc dấu để học tín hiệu ngôn ngữ.
- Các cách phiên âm đã biết trong Popular Names được nối thành nhóm trước khi chia tập. Những cách viết tương đương chưa có trong nguồn vẫn có thể bị bỏ sót; đây không phải kiểm chứng mọi bí danh của mọi tên.
- Nếu các nguồn gán nhiều nhóm cho cùng khóa hoặc nhóm phiên âm, loại toàn bộ nhóm khỏi train/test chính; lưu trong `data/ambiguous.csv`. Không ngẫu nhiên chọn một nhãn cho `Lee`.
- Chia 70/15/15 theo nhóm và phân tầng nhãn. Tên có một nhãn nghĩa là một nhãn **trong 10 nhóm đã chọn**, không khẳng định nó chỉ tồn tại trong một ngôn ngữ trên thế giới.
- Vocabulary chỉ học từ train. Bản bỏ dấu chỉ được bổ sung **trong train**, không tạo thêm biến thể để nhân số test.
- Cross-entropy có trọng số căn bậc hai nghịch tần suất để giảm ảnh hưởng mất cân bằng; gradient clipping và AdamW.
- Chọn epoch và temperature bằng validation. Baseline TF-IDF ký tự + Logistic Regression dùng cùng cách chia, chọn C bằng validation.
- Báo cả accuracy, balanced accuracy, Macro-F1, top-3 accuracy và ma trận nhầm lẫn. Tách accuracy của phần test từ điển/tutorial và Wikidata-only.

Số test Việt chỉ 68; không suy rộng một kết quả trên tập nhỏ thành độ chính xác cho mọi tên Việt. Bảng nhãn Wikidata có thể chứa tên của người nhập cư hoặc lỗi trích xuất. Ngưỡng 0,95 là liên hệ trong cohort sáu nhóm của nguồn, không phải chứng nhận nguồn gốc hay mức tin cậy của LSTM.

## Các tệp chính

- `name_origin/prepare.py`: tải, kiểm tra checksum, lọc nhãn, gộp biến thể, chia dataset.
- `name_origin/model.py`: Embedding và LSTM.
- `name_origin/train.py`: huấn luyện, hiệu chỉnh điểm, baseline, xuất đánh giá và biểu đồ.
- `name_origin/predict.py`: CLI, dự đoán tên, cảnh báo tên nhiều nhãn, gộp thành phần họ tên.
- `app.py`, `templates/`, `static/`: demo Flask tiếng Anh.
- `artifacts/model.pt`: checkpoint đã train.
- `artifacts/metrics.json`: số đo chi tiết; `test_predictions.csv`: dự đoán từng tên test để tự kiểm tra.
- `artifacts/confusion_matrix.png`, `training_curves.png`: hình dùng trong báo cáo.
- `docs/KET_QUA.md`: kết quả thực tế của bản cuối.
- `docs/THUYET_TRINH.md`: dàn ý trình bày và câu hỏi thường gặp.
- `Name_Origin_LSTM.ipynb`: notebook gọi các bước của dự án.

Các thư mục `artifacts/three_sources` và `artifacts/ascii_only` giữ kết quả thử nghiệm trước. Dataset/split hoặc cách biểu diễn input khác nhau; không dùng các con số đó để tuyên bố so sánh đối chứng trên cùng benchmark.

## Nguồn và giấy phép

Giữ thông tin tác giả và liên kết nguồn khi dùng trong đồ án. Vietnamese NameDB ghi MIT; Popular Names ghi CC0, bản thông báo giấy phép đã lưu ở `data/raw/`. Behind the Name và các tệp dữ liệu khác không tự động được cấp giấy phép bởi mã mới của dự án; tham khảo điều khoản nguồn trước khi phân phối lại công khai. Manifest ghi provenance cho việc tái tạo bài tập.

Kiểm tra của dự án bao gồm checksum, không trùng tên/nhóm giữa các tập, biến thể phiên âm, khớp số đo với CSV dự đoán, padding đúng, cảnh báo `Lee`, gộp thành phần họ tên và API nhận input không hợp lệ.
