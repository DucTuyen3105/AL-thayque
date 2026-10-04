# Kết quả mô hình LSTM đã chạy

Các số dưới đây được sinh từ `artifacts/metrics.json`, không phải mức chính xác dự kiến.

Checkpoint: epoch 15; 87,114 tham số; temperature 0.95; thêm 384 biến thể bỏ dấu trong train.

| Chỉ số | LSTM | TF-IDF ký tự + Logistic Regression |
|---|---:|---:|
| Accuracy | 83.24% | 83.09% |
| Balanced accuracy | 69.52% | 70.08% |
| Macro-F1 | 67.79% | 67.63% |
| Top-3 accuracy | 93.52% | 93.48% |

Baseline luôn đoán nhóm đông nhất: 30.79% accuracy.

## Điểm từng nhóm trên test

| Nhóm | Mẫu test | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Chinese | 802 | 86.87% | 80.05% | 83.32% |
| English | 844 | 67.83% | 75.71% | 71.56% |
| French | 89 | 46.15% | 53.93% | 49.74% |
| German | 151 | 43.16% | 54.30% | 48.09% |
| Italian | 176 | 59.51% | 69.32% | 64.04% |
| Japanese | 1890 | 93.67% | 90.11% | 91.86% |
| Korean | 573 | 89.72% | 85.34% | 87.48% |
| Russian | 1442 | 93.53% | 90.29% | 91.88% |
| Spanish | 103 | 36.03% | 47.57% | 41.00% |
| Vietnamese | 68 | 49.25% | 48.53% | 48.89% |

## Phạm vi và cách diễn giải

- Test chính có 6,138 tên. Dữ liệu gồm cả nhãn Wikidata suy ra; đây không phải xác minh ngôn ngữ hay từ nguyên của người mang tên.
- Phần test có nguồn từ điển/tutorial: 3,059 tên, accuracy 81.07%.
- Phần chỉ có nhãn Wikidata: 3,053 tên, accuracy 85.62%.
- Cùng test khi bỏ mọi dấu: accuracy 83.15%, Macro-F1 67.23%.
- Riêng Vietnamese: F1 giữ dấu 48.89%; bỏ dấu 46.62%. Chỉ có 68 mẫu test, nên chưa đủ để khẳng định độ chính xác cao cho tên Việt nói chung.
- Điểm các nhóm nhỏ và tên dùng chung còn thấp. Tập chính đã loại tên xung đột, nên khó khăn thực tế có thể cao hơn.
- Các bản thử nghiệm đã được xem nhiều lần trong quá trình phát triển. Muốn một benchmark độc lập cuối cùng, cần thêm test ngoài nguồn chưa từng xem.
- Với họ tên đầy đủ, trung bình điểm thành phần là heuristic phục vụ demo; chưa có benchmark họ tên đầy đủ.

## Ví dụ chạy thực tế

| Input | Nhóm có điểm cao nhất | Điểm mô hình | Có cảnh báo mơ hồ |
|---|---|---:|---|
| Nguyễn | Vietnamese | 95.47% | Không |
| Nguyen | Vietnamese | 99.28% | Không |
| Nguyễn Đức Tuyên | Vietnamese | 92.06% | Không |
| Nguyen Duc Tuyen | Vietnamese | 84.76% | Không |
| Yamamoto | Japanese | 96.54% | Không |
| Kim Min-jun | Korean | 58.40% | Có |
| John Smith | English | 59.23% | Không |
| Pierre Martin | French | 60.46% | Có |
| Lee | English | 96.16% | Có |

Ví dụ demo có thể là tên đã có trong train; không coi các ví dụ này là bằng chứng tổng quát hóa. Phần trăm không phải xác suất quốc tịch của một người.

## Hình và dữ liệu kiểm chứng

- `artifacts/confusion_matrix.png`: ma trận nhầm lẫn; ô ghi số mẫu, màu thể hiện tỷ lệ theo nhãn thật.
- `artifacts/training_curves.png`: loss train và điểm validation.
- `artifacts/test_predictions.csv`: tên, nhãn nguồn, dự đoán, điểm và kết quả đúng/sai của mỗi tên test.
- `artifacts/metrics.json`: toàn bộ số liệu và manifest nguồn.
- `artifacts/demo_predictions.json`: dự đoán mẫu thực tế, gồm cảnh báo và điểm từng thành phần.
