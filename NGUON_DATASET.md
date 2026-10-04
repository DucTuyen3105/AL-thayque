# Nguồn dataset cho bài toán phân loại nhóm ngôn ngữ của tên bằng LSTM

Tài liệu này ghi lại **năm nguồn dữ liệu đã tải và thực sự dùng trong dự án**. Các số lượng được đối chiếu với tệp trong `data/raw/`, [thống kê xử lý](data/stats.json) và [manifest nguồn](data/raw/manifest.json).

Dữ liệu dùng để học đặc trưng của họ, tên riêng và thành phần tên trong 10 nhóm: Chinese, English, French, German, Italian, Japanese, Korean, Russian, Spanish và Vietnamese. Nhãn của dataset không xác định quốc tịch, dân tộc hay ngôn ngữ một người sử dụng.

## 1. Tổng quan các nguồn

| Nguồn | Loại dữ liệu | Số dòng trong bản gốc đã tải | Lượt tên–nhãn được nhận trước khi gộp | Tên cuối có ghi nhận nguồn này |
|---|---|---:|---:|---:|
| PyTorch Names | Danh sách họ theo nhóm ngôn ngữ | 20.074, thuộc 18 nhóm | 16.506 | 15.143 |
| Behind the Name / Anwarvic | Tên riêng và trường nguồn gốc | 20.505 | 8.557 | 5.612 |
| Popular Names by Country / sigpwned | Tên riêng, họ, quốc gia và cách viết | 5.056 | 911 | 456 |
| Vietnamese NameDB / duyet | Tên riêng tiếng Việt, gồm tên ghép | 2.552 | 432 | 231 |
| Wikidata / Stanford RegLab | Bảng liên hệ tên với sáu nhóm châu Á | 44.579 | 24.794 | 21.529 |

**Cách đọc số lượng:**

- Số dòng gốc là số đo từ bản đã tải, có thể khác số tên giới thiệu trên trang nguồn.
- “Lượt tên–nhãn được nhận” là số lượt đi qua bộ lọc của từng nguồn. Một tên có thể được nhận nhiều lần hoặc đi kèm nhiều nhãn; đây chưa phải số tên độc lập.
- Cột cuối đếm tên trong `data/names.csv` có nguồn tương ứng. Một tên có thể thuộc nhiều nguồn, nên **không cộng cột này để suy ra tổng dataset**.
- Tổng số tên sau gộp và loại nhãn xung đột là **40.888**. Trong đó, **1.825 tên** có ghi nhận từ ít nhất hai nguồn.

## 2. Chi tiết từng nguồn

### 2.1. PyTorch Names

**Nguồn:** tutorial chính thức [NLP From Scratch: Classifying Names with a Character-Level RNN](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html).

**File tải:** [data.zip](https://download.pytorch.org/tutorial/data.zip), lưu thành [pytorch_names.zip](data/raw/pytorch_names.zip).

Trong file ZIP, thư mục `data/names/` chứa 18 file dạng `[Language].txt`, mỗi dòng là một tên. Dự án chỉ đọc 10 file tương ứng với 10 nhóm được chọn. Nhãn lấy từ tên file, chẳng hạn `Japanese.txt` → `Japanese`.

Đếm trực tiếp được 20.074 dòng ở cả 18 file. Riêng 10 nhóm đã chọn có 16.510 dòng; 4 dòng không đạt điều kiện ký tự/độ dài, còn 16.506 lượt được nhận trước bước gộp toàn bộ nguồn.

**Vai trò:** cung cấp dữ liệu họ và một bộ dữ liệu nền dễ kiểm tra. Dự án dùng dữ liệu của tutorial nhưng tự xây dựng mô hình LSTM.

**Hạn chế:** mất cân bằng lớn; bản gốc có 9.408 dòng Russian, 94 Korean và 73 Vietnamese. Đây không phải bộ họ tên đầy đủ, cũng không phải khảo sát đại diện dân số. Tutorial mô tả cách sử dụng; lịch sử thu thập từng tên không có trong file ZIP.

### 2.2. Behind the Name / Anwarvic

**Kho cung cấp:** [Anwarvic/Behind-The-Name](https://github.com/Anwarvic/Behind-The-Name). Tác giả kho mô tả dữ liệu được lấy từ website [Behind the Name](https://www.behindthename.com/).

**File tải:** [dataset.csv](https://raw.githubusercontent.com/Anwarvic/Behind-The-Name/master/dataset.csv), lưu thành [behind_name.csv](data/raw/behind_name.csv), gồm **20.505 dòng**.

Dự án đọc hai trường `name` và `origin`. Chỉ nhận nhãn hiện đại khớp 10 nhóm mục tiêu; không tự đổi `Ancient Germanic` thành `German`. Các ghi chú trong ngoặc của nhãn được bỏ; số phân biệt cuối tên như `(1)` được bỏ khỏi nội dung tên.

Một tên có thể có nhiều nhãn, nên 8.557 lượt tên–nhãn được nhận không có nghĩa là 8.557 tên khác nhau. Xung đột nhãn được xử lý sau khi gộp với các nguồn còn lại.

**Vai trò:** bổ sung tên riêng, giúp mô hình học cả họ lẫn tên.

**Hạn chế:** tên có thể phổ biến trong nhiều ngôn ngữ; `origin` không phải quốc tịch người mang tên. Bản dùng là dữ liệu tải từ kho công khai, không phải một lần thu thập mới trực tiếp từ website. Kho không công bố giấy phép dữ liệu riêng rõ ràng trong phần đã đối chiếu.

### 2.3. Popular Names by Country / sigpwned

**Nguồn:** [sigpwned/popular-names-by-country-dataset](https://github.com/sigpwned/popular-names-by-country-dataset). Theo tác giả, dữ liệu được tổng hợp từ danh sách tên phổ biến trên Wikipedia và có hỗ trợ cách viết Latin/phiên âm.

Hai file đã tải:

| File nguồn | File trong dự án | Số dòng |
|---|---|---:|
| [common-forenames-by-country.csv](https://raw.githubusercontent.com/sigpwned/popular-names-by-country-dataset/main/common-forenames-by-country.csv) | [popular_common-forenames-by-country.csv](data/raw/popular_common-forenames-by-country.csv) | 2.480 |
| [common-surnames-by-country.csv](https://raw.githubusercontent.com/sigpwned/popular-names-by-country-dataset/main/common-surnames-by-country.csv) | [popular_common-surnames-by-country.csv](data/raw/popular_common-surnames-by-country.csv) | 2.576 |

Dự án ánh xạ quốc gia đại diện sang nhóm tên như sau:

| Mã quốc gia | Nhãn trong dự án |
|---|---|
| CN | Chinese |
| GB | English |
| FR | French |
| DE | German |
| IT | Italian |
| JP | Japanese |
| KR | Korean |
| RU | Russian |
| ES | Spanish |
| VN | Vietnamese |

Đọc `Romanized Name` và `Localized Name`, chỉ nhận cách viết hợp lệ bằng chữ Latin. Các biến thể chung `Name Group` được nối thành một nhóm trước khi chia train/test. Có 911 lượt tên–nhãn được nhận trước khi gộp.

**Vai trò:** bổ sung họ phổ biến và thông tin về cách viết tương đương.

**Hạn chế:** quốc gia nơi tên phổ biến là nhãn thay thế, không chứng minh nguồn gốc ngôn ngữ. Quy tắc `GB → English` là lựa chọn của dự án, không khẳng định mọi tên ở Anh đều có nguồn gốc tiếng Anh.

Nguồn ghi giấy phép **CC0**; thông báo được lưu tại [popular_LICENSE](data/raw/popular_LICENSE).

### 2.4. Vietnamese NameDB / duyet

**Nguồn:** [duyet/vietnamese-namedb](https://github.com/duyet/vietnamese-namedb), từ điển tên Việt cộng đồng do Van-Duyet Le duy trì.

Hai file gốc đã dùng:

| File nguồn | File trong dự án | Số dòng |
|---|---|---:|
| [boy.txt](https://raw.githubusercontent.com/duyet/vietnamese-namedb/master/boy.txt) | [vietnamese_boy.txt](data/raw/vietnamese_boy.txt) | 1.236 |
| [girl.txt](https://raw.githubusercontent.com/duyet/vietnamese-namedb/master/girl.txt) | [vietnamese_girl.txt](data/raw/vietnamese_girl.txt) | 1.316 |

Các file chứa tên ghép như `Anh Dũng`, `Bảo An`. Dự án tách theo dấu cách và gộp các thành phần trùng: ví dụ `Anh Dũng` → `Anh`, `Dũng`. Có **455 thành phần khác nhau theo văn bản gốc**; bộ lọc nhận 432 lượt hợp lệ, loại 23 lượt không hợp lệ.

**Vai trò:** tăng độ phủ của thành phần tên Việt. Nhãn `Vietnamese` được gán theo phạm vi của từ điển; thông tin giới tính không dùng làm đầu vào hoặc mục tiêu học.

**Hạn chế:** 2.552 dòng tên ghép không tương đương 2.552 thành phần độc lập. Dự án không ghép ngẫu nhiên họ và tên để tăng số mẫu. Đây là dữ liệu cộng đồng, không phải thống kê dân số hay danh sách đã được kiểm chứng toàn bộ.

Kho ghi giấy phép **MIT**; bản thông báo được lưu tại [vietnamese_LICENSE](data/raw/vietnamese_LICENSE).

### 2.5. Wikidata / Stanford RegLab

**Nguồn dữ liệu:** [reglab/disaggregation](https://github.com/reglab/disaggregation).

**Nghiên cứu liên quan:** Lin và cộng sự, *Enabling disaggregation of Asian American subgroups: a dataset of Wikidata names for disparity estimation*, Scientific Data, 2025. [Bài báo và DOI](https://doi.org/10.1038/s41597-025-04753-y).

Hai bảng đã tải:

| File nguồn | File trong dự án | Số dòng |
|---|---|---:|
| [p_race_given_firstname.csv](https://raw.githubusercontent.com/reglab/disaggregation/main/data/name_race_data/p_race_given_firstname.csv) | [wikidata_given.csv](data/raw/wikidata_given.csv) | 25.876 |
| [p_race_given_surname.csv](https://raw.githubusercontent.com/reglab/disaggregation/main/data/name_race_data/p_race_given_surname.csv) | [wikidata_surname.csv](data/raw/wikidata_surname.csv) | 18.703 |

Theo bài báo, các bảng được xây dựng từ mẫu Wikidata hơn 300.000 cá nhân, với sáu nhóm Asian Indian, Chinese, Filipino, Japanese, Korean và Vietnamese. **Số cá nhân không phải số tên độc lập.** Nghiên cứu sử dụng nơi sinh/quốc tịch làm thông tin thay thế khi xây dựng bảng liên hệ tên–nhóm.

**Cách dùng trong dự án:** tìm nhóm có giá trị cao nhất trên cả sáu cột; chỉ nhận nếu nhóm đó thuộc Chinese, Japanese, Korean hoặc Vietnamese và giá trị ≥ **0,95**. Không chuẩn hóa lại bảng sau khi bỏ hai nhóm còn lại. Có 24.794 lượt tên–nhãn được nhận; mã nguồn đánh dấu bằng `wikidata_silver`.

**Hạn chế:** đây là **nhãn suy ra**, không phải nhãn ngôn ngữ hay từ nguyên do chuyên gia xác minh. Ngưỡng 0,95 của nguồn không phải độ chính xác 95% của LSTM. Dự án chỉ dùng hai bảng tên công khai, không dùng dữ liệu bệnh nhân hoặc địa lý của nghiên cứu.

## 3. Cách hợp nhất các nguồn

Mã thực hiện nằm trong [name_origin/prepare.py](name_origin/prepare.py).

1. Đọc từng nguồn, áp dụng quy tắc chọn nhãn riêng của nguồn.
2. Tạo khóa tên bằng chữ thường và bỏ dấu để kiểm tra trùng: `Nguyễn`, `NGUYỄN`, `Nguyen` có cùng khóa. Nội dung đưa vào LSTM vẫn giữ dấu Latin.
3. Gộp các tên cùng khóa và nối các cách viết tương đương đã biết qua `Name Group`.
4. Tổng hợp mọi nhãn của từng nhóm. Nếu có nhiều nhãn trong 10 nhóm đã chọn, loại nhóm khỏi dataset chính và lưu vào [ambiguous.csv](data/ambiguous.csv).
5. Chia nhóm tên theo tỷ lệ 70/15/15, phân tầng nhãn, seed 42. Một nhóm không được nằm ở nhiều tập.

Kết quả kiểm tra từ dữ liệu dự án:

| Chỉ tiêu | Số lượng |
|---|---:|
| Lượt tên–nhãn được nhận từ năm nguồn | 51.200 |
| Tên chuẩn hóa khác nhau trước khi loại xung đột | 43.257 |
| Tên bị loại do xung đột nhãn trong nhóm | 2.369 |
| Tên cuối trong dataset chính | 40.888 |
| Nhóm độc lập dùng để chia tập | 40.848 |
| Train | 28.611 |
| Validation | 6.139 |
| Test | 6.138 |

Một nhãn duy nhất ở đây nghĩa là một nhãn trong **10 nhóm đã chọn**, không có nghĩa tên đó chỉ tồn tại trong một ngôn ngữ trên thế giới. Các cách viết tương đương không được ghi trong nguồn vẫn có thể bị bỏ sót.

## 4. Phân bố dataset cuối

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
| **Tổng** | **40.888** | **28.611** | **6.139** | **6.138** |

Dataset vẫn mất cân bằng và còn lỗi nhãn tiềm tàng. Vì vậy cần xem Macro-F1 và điểm từng nhóm; không dùng riêng accuracy tổng để kết luận mọi nhóm đều được nhận diện tốt. Kết quả mô hình thực tế nằm trong [docs/KET_QUA.md](docs/KET_QUA.md).

## 5. Kiểm tra phiên bản và tái tạo dữ liệu

[data/raw/manifest.json](data/raw/manifest.json) ghi **URL tải, kích thước byte và SHA-256** cho cả tám file đầu vào. Các URL GitHub dùng nhánh `main`/`master`, nên nội dung trên mạng có thể thay đổi; checksum xác định bản thực tế đang được dùng trong dự án.

Với dữ liệu raw đã có sẵn, tái tạo dataset bằng:

```bash
source .venv/bin/activate
python -m name_origin.prepare --offline
```

Các file để kiểm tra:

- [data/names.csv](data/names.csv): dataset đã hợp nhất; trường `sources` ghi tất cả nguồn của từng tên.
- [data/train.csv](data/train.csv), [data/val.csv](data/val.csv), [data/test.csv](data/test.csv): ba tập đã chia.
- [data/ambiguous.csv](data/ambiguous.csv): tên/nhóm bị loại vì nhiều nhãn.
- [data/stats.json](data/stats.json): số lượng và phân bố lớp thực tế.
- [data/raw/manifest.json](data/raw/manifest.json): danh sách file nguồn và checksum.

Khi trích dẫn trong báo cáo, ghi tên kho/tổ chức, đường dẫn gốc và bài báo RegLab. Giấy phép ghi trong một kho hoặc bài báo không tự động áp dụng cho mọi nguồn khác được gộp vào dataset này; thông tin bản quyền của mỗi nguồn được giữ riêng.
