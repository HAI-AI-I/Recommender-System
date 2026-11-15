# 🎬 Hệ Thống Gợi Ý Phim Hybrid AI (MovieLens 32M)
## Kết hợp Collaborative Filtering (CF) & Machine Learning (LightGBM Ranker)

Dự án này được chuyển đổi trực tiếp từ Google Colab Notebook (`MovieLens_32M_Hybrid_RecSys.ipynb`) để chạy độc lập trên môi trường máy tính cục bộ.

---

### 📁 Cấu trúc thư mục dự án:

```text
D:\learn_ai\project\Recommender-System\
│
├── MovieLens_32M_Hybrid_RecSys.ipynb  # Bản sao lưu của Jupyter Notebook trên Google Colab
├── phase1_data_preprocessing.py       # Code Python Giai đoạn 1 (tiền xử lý dữ liệu)
├── phase2_collaborative_filtering.py  # Code Python Giai đoạn 2 (CF lọc 300 ứng viên)
├── phase3_ranking_model.py            # Code Python Giai đoạn 3 (LightGBM chấm điểm & xếp hạng)
├── README.md                          # Hướng dẫn chi tiết dự án
│
└── dataset\                           # Thư mục chứa toàn bộ dữ liệu thô và dữ liệu đã chế biến
    ├── ml-32m.zip                     # Tập dữ liệu gốc MovieLens 32M (238 MB)
    ├── movies_processed.csv.gz        # 87,585 phim với 55 đặc trưng (năm, 18 thể loại, 30 tags TF-IDF)
    ├── movie_stats.csv.gz             # Thống kê điểm số và độ phổ biến của từng phim
    ├── user_stats.csv.gz              # Thống kê gu chấm điểm của 200,948 người dùng
    ├── train_interactions.csv.gz      # 12,063,020 tương tác huấn luyện (nhãn y = 0/1)
    ├── test_interactions.csv.gz       # 3,005,773 tương tác kiểm định theo dòng thời gian
    ├── user_latent.npy                # Ma trận tiềm ẩn người dùng (20,000 x 32)
    ├── movie_latent.npy               # Ma trận tiềm ẩn bộ phim (68,357 x 32)
    ├── cf_mappings.pkl                # Bộ từ điển ánh xạ ID sang chỉ số ma trận
    ├── lgbm_ranker.txt                # Mô hình LightGBM Ranker đã huấn luyện
    └── feature_cols.pkl               # Danh sách 57 cột đặc trưng của mô hình
```

---

### ⚙️ Cách chạy mã nguồn offline:

1. **Cài đặt thư viện:**
   ```bash
   pip install pandas numpy scikit-learn lightgbm scipy matplotlib
   ```

2. **Chạy tiền xử lý dữ liệu (Giai đoạn 1):**
   ```bash
   python phase1_data_preprocessing.py
   ```

3. **Chạy Collaborative Filtering (Giai đoạn 2):**
   ```bash
   python phase2_collaborative_filtering.py
   ```

4. **Chạy Mô hình Xếp hạng LightGBM (Giai đoạn 3):**
   ```bash
   python phase3_ranking_model.py
   ```

---

### 📌 Lộ trình 4 giai đoạn:
* [x] **Giai đoạn 1:** Chuẩn bị & Chế biến dữ liệu (Feature Engineering) - *Hoàn tất*.
* [x] **Giai đoạn 2:** Lọc nhanh ứng viên (Candidate Retrieval via Collaborative Filtering - Top 300) - *Hoàn tất*.
* [x] **Giai đoạn 3:** Chấm điểm & Xếp hạng chi tiết (Scoring & Re-ranking via LightGBM) - *Hoàn tất*.
* [ ] **Giai đoạn 4:** Demo gợi ý phim End-to-End & Giao diện tương tác người dùng.
