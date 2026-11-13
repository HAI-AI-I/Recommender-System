# 🎬 Hệ Thống Gợi Ý Phim Hybrid AI (MovieLens 32M)
## Kết hợp Collaborative Filtering (CF) & Machine Learning (LightGBM Ranker)

Dự án này được chuyển đổi trực tiếp từ Google Colab Notebook (`MovieLens_32M_Hybrid_RecSys.ipynb`) để chạy độc lập trên môi trường máy tính cục bộ.

---

### 📁 Cấu trúc thư mục dự án:

```text
D:\learn_ai\project\Recommender-System\
│
├── phase1_data_preprocessing.py   # Code Python Giai đoạn 1 (tương đương với các cell trên Colab)
├── README.md                      # Hướng dẫn chi tiết dự án
│
└── dataset\                       # Thư mục chứa toàn bộ dữ liệu thô và dữ liệu đã chế biến
    ├── ml-32m.zip                 # Tập dữ liệu gốc MovieLens 32M (238 MB)
    ├── movies_processed.csv.gz    # 87,585 phim với 55 đặc trưng (năm, 18 thể loại, 30 tags TF-IDF)
    ├── movie_stats.csv.gz         # Thống kê điểm số và độ phổ biến của từng phim
    ├── user_stats.csv.gz          # Thống kê gu chấm điểm của 200,948 người dùng
    ├── train_interactions.csv.gz  # 12,063,020 tương tác huấn luyện (nhãn y = 0/1)
    └── test_interactions.csv.gz   # 3,005,773 tương tác kiểm định theo dòng thời gian
```

---

### Cách chạy mã nguồn:

1. **Cài đặt thư viện cần thiết:**
   ```bash
   pip install pandas numpy scikit-learn lightgbm scipy
   ```

2. **Chạy tiền xử lý dữ liệu Giai đoạn 1:**
   ```bash
   python phase1_data_preprocessing.py
   ```
   *(Toàn bộ các tệp đặc trưng đã được tạo sẵn trong thư mục `dataset/`, bạn có thể chạy lại script này bất cứ lúc nào để tái tạo dữ liệu).*

---

### Lộ trình 4 giai đoạn:
* [x] **Giai đoạn 1:** Chuẩn bị & Chế biến dữ liệu (Feature Engineering) - *Đã hoàn tất*.
* [ ] **Giai đoạn 2:** Lọc nhanh ứng viên (Candidate Retrieval via Collaborative Filtering - Top 300).
* [ ] **Giai đoạn 3:** Chấm điểm & Xếp hạng chi tiết (Scoring & Re-ranking via LightGBM).
* [ ] **Giai đoạn 4:** Demo gợi ý phim End-to-End & Đánh giá mô hình.
