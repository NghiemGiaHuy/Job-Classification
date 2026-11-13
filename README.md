# Job Classification

Dự án thử nghiệm phân loại cấp bậc nghề nghiệp (`career_level`) từ thông tin trong tin tuyển dụng. Mã nguồn hiện tại so sánh hai mô hình học máy: **Logistic Regression** và **Linear SVM**.

## Dữ liệu đầu vào

Tệp dữ liệu được lưu trong repository dưới tên `final_project.ods`. Script đọc tệp này và sử dụng các cột:

| Cột | Vai trò |
|---|---|
| `title` | Tiêu đề tin tuyển dụng |
| `description` | Mô tả công việc |
| `function` | Nhóm chức năng |
| `industry` | Ngành nghề |
| `career_level` | Nhãn cấp bậc cần dự đoán (target) |

Các dòng trùng lặp và dòng có giá trị thiếu sẽ bị loại trước khi huấn luyện. Dữ liệu được chia thành tập train/test theo tỷ lệ 80/20, có phân tầng theo nhãn.

## Cách xử lý và huấn luyện

- Dùng TF-IDF để biểu diễn `title` và `description` thành đặc trưng văn bản.
- Dùng One-Hot Encoding cho `function` và `industry`.
- Áp dụng SMOTEN trên tập huấn luyện để xử lý mất cân bằng nhãn.
- Huấn luyện và đánh giá Logistic Regression cùng Linear SVM.
- In Accuracy, Macro F1, Weighted F1, báo cáo phân loại và bảng so sánh mô hình.

## Yêu cầu

- Python 3.9 trở lên (khuyến nghị)
- Các thư viện Python: `pandas`, `matplotlib`, `seaborn`, `imbalanced-learn`, `scikit-learn`, `odfpy`

## Cài đặt và chạy

Clone repository, sau đó cài các thư viện cần thiết:

```bash
git clone https://github.com/NghiemGiaHuy/Job-Classification.git
cd Job-Classification
python -m venv .venv
```

Kích hoạt môi trường ảo:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Cài dependencies và chạy chương trình:

```bash
python -m pip install pandas matplotlib seaborn imbalanced-learn scikit-learn odfpy
python job_cls.py
```

Đảm bảo `final_project.ods` nằm cùng thư mục với `job_cls.py` khi chạy.

## Kết quả tham khảo

Trong phần kết quả mẫu được ghi chú trong mã nguồn:

| Mô hình | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.7654 | 0.6954 | 0.7680 |
| Linear SVM | 0.7548 | 0.4974 | 0.7516 |

Đây là số liệu tham khảo từ lần chạy trước; kết quả thực tế có thể khác tùy phiên bản thư viện và dữ liệu.

## Cấu trúc repository

```text
Job-Classification/
├── final_project.ods   # Dữ liệu tuyển dụng
└── job_cls.py           # Tiền xử lý, huấn luyện và đánh giá mô hình
```

## Lưu ý

- Tên nhãn và số lượng mẫu được SMOTEN tăng cường hiện được khai báo trực tiếp trong `job_cls.py`. Nếu dữ liệu thay đổi, hãy kiểm tra cấu hình `sampling_strategy` để bảo đảm các nhãn đó tồn tại và số lượng mẫu mục tiêu phù hợp.
- Biểu đồ phân bố nhãn và confusion matrix hiện đang được comment trong mã nguồn.

## Giấy phép

Repository hiện chưa nêu giấy phép sử dụng. Hãy bổ sung giấy phép nếu muốn quy định rõ cách người khác có thể sử dụng hoặc phân phối mã nguồn và dữ liệu.
