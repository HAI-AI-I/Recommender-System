"""
Hệ Thống Gợi Ý Phim Hybrid AI (MovieLens 32M)
================================================================================
Quy trình kỹ thuật gồm 4 giai đoạn tích hợp:
- GIAI ĐOẠN 1: Chuẩn bị & Tiền xử lý dữ liệu (Feature Engineering, TF-IDF Tags, Điền khuyết 3 tầng, Temporal Split)
- GIAI ĐOẠN 2: Lọc nhanh 300 ứng viên tiềm năng (Candidate Retrieval bằng TruncatedSVD Matrix Factorization)
- GIAI ĐOẠN 3: Chấm điểm & Xếp hạng lại bằng Machine Learning (LightGBM Classifier + Hybrid Negative Sampling)
- GIAI ĐOẠN 4: Đóng gói Class HybridMovieRecommender, giải quyết Cold-Start & Đánh giá End-to-End (Vòng 1 vs Vòng 2)
================================================================================
"""

import os
import re
import sys
import time
import json
import random
import pickle
import zipfile
import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb


# ==============================================================================
# 0. CẤU HÌNH THƯ MỤC & THAM SỐ TOÀN CỤC
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "dataset"
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"


def log_step(title):
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


# 1 TIỀN XỬ LÝ DỮ LIỆU & COLLABORATIVE FILTERING (PHASE 1 & 2)
def check_and_download_dataset(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-32m.zip"

    if not zip_path.exists():
        print(f"Chưa có dataset")
        urllib.request.urlretrieve(ZIP_URL, zip_path)
        print("Tải dataset thành công về:", zip_path)
    return zip_path


def process_phase1_data(zip_path, n_users=20000):
    """
    Thực hiện Giai đoạn 1:
    - Xử lý phim, trích xuất năm (Regex), mã hóa One-hot thể loại.
    - Trích xuất 30 đặc trưng Tags bằng TF-IDF kết hợp L2-norm.
    - Lọc 20,000 người dùng tích cực nhất (ratings >= 20).
    - Phân chia Temporal Split 80/20 theo thời gian đánh giá của từng user.
    - Thống kê phim & người dùng chỉ trên tập Train (Triệt tiêu 100% Data Leakage).
    - Điền khuyết release_year theo chiến lược 3 tầng.
    """
    log_step("GIAI ĐOẠN 1: Chuẩn bị & Chế biến dữ liệu (Feature Engineering)")

    with zipfile.ZipFile(zip_path, 'r') as z:
        # 1 Xử lý movies.csv
        print("1.1 Đọc và làm sạch movies.csv...")
        with z.open('ml-32m/movies.csv') as f:
            df_movies = pd.read_csv(f)

        # Lấy năm phát hành (Regex Tầng 1)
        df_movies['release_year'] = df_movies['title'].str.extract(r'\((\d{4})\)').astype(float)
        df_movies['clean_title'] = df_movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()

        # Lấy danh sách thể loại từ dataset
        all_genres = sorted(
            set(g for genres in df_movies['genres'].dropna().str.split('|') for g in genres)
            - {'(no genres listed)', 'IMAX'}
        )
        print(f"Số lượng thể loại là: {len(all_genres)}")
        print(f"{all_genres}")

        # One-hot encoding cho từng thể loại
        for g in all_genres:
            pattern = r'Children' if g == 'Children' else g
            df_movies[f'genre_{g}'] = df_movies['genres'].str.contains(pattern, regex=True).fillna(False).astype(int)
        df_movies['genre_count'] = df_movies[[f'genre_{g}' for g in all_genres]].sum(axis=1)

        # 1.2. Xử lý tags.csv bằng TF-IDF
        print("1.2 Đọc tags.csv và trích xuất 30 đặc trưng TF-IDF")
        with z.open('ml-32m/tags.csv') as f:
            df_tags = pd.read_csv(f, usecols=['movieId', 'tag'])

        df_tags = df_tags.dropna(subset=['tag'])
        df_tags['tag'] = df_tags['tag'].astype(str).str.lower().str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True).str.strip()
        movie_tags = df_tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.astype(str))).reset_index()

        tfidf = TfidfVectorizer(max_features=30, stop_words='english')
        tag_features = tfidf.fit_transform(movie_tags['tag']).toarray()
        feature_names = [f'tag_{w}' for w in tfidf.get_feature_names_out()]
        df_tag_features = pd.DataFrame(tag_features, columns=feature_names)
        df_tag_features['movieId'] = movie_tags['movieId']

        # Gộp bảng phim hoàn chỉnh      
        df_movies_final = df_movies.merge(df_tag_features, on='movieId', how='left')
        for col in feature_names:
            df_movies_final[col] = df_movies_final[col].fillna(0.0)

        # 1.3 Lấy mẫu 20,000 người dùng tích cực nhất từ ratings.csv bằng Counter & Chunk
        from collections import Counter
        chunk_size = 2_000_000
        user_activity = Counter()

        print("1.3 Quét ratings.csv để đếm lượt đánh giá của từng user...")
        with z.open('ml-32m/ratings.csv') as f:
            reader = pd.read_csv(f, chunksize=chunk_size, usecols=['userId'], dtype={'userId': np.int32})
            for chunk in reader:
                user_activity.update(chunk['userId'])

        top_users = [u for u, count in user_activity.most_common() if count >= 30][:n_users]
        top_users_set = set(top_users)
        print(f"Đã chọn {len(top_users):,} Core Users")
        print(f'Lượt đánh giá nhiều nhất: {user_activity[top_users[0]]:,}')
        print(f'Lượt đánh giá ít nhất: {user_activity[top_users[-1]]:,}')
        
        print(f"1.4 Lọc tương tác của {len(top_users):,} Core Users theo từng khối")
        sampled_chunks = []
        with z.open('ml-32m/ratings.csv') as f:
            reader = pd.read_csv(f, chunksize=chunk_size, dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32, 'timestamp': np.int64})
            for chunk in reader:
                chunk_filtered = chunk[chunk['userId'].isin(top_users_set)]
                if len(chunk_filtered) > 0:
                    sampled_chunks.append(chunk_filtered)

        df_sample = pd.concat(sampled_chunks, ignore_index=True)
        df_sample['target'] = (df_sample['rating'] >= 4.0).astype(np.int8)

        # 1.5. Temporal Split: 80% Train, 20% Test theo thứ tự thời gian (Vectorized cumcount)
        print("1.5 Phân chia tập Train / Test theo thời gian (Temporal Split 80/20)")
        df_sample = df_sample.sort_values(['userId', 'timestamp']).reset_index(drop=True)
        df_sample['rank_pct'] = df_sample.groupby('userId').cumcount() / df_sample.groupby('userId')['timestamp'].transform('count')
        train_mask = df_sample['rank_pct'] < 0.8
        df_train = df_sample[train_mask].drop(columns=['rank_pct']).copy()
        df_test = df_sample[~train_mask].drop(columns=['rank_pct']).copy()
        df_sample = df_sample.drop(columns=['rank_pct'])

        # 1.5. Thống kê đặc trưng phim & người dùng trên tập train
        print("1.5 Tính toán thống kê phim và người dùng chỉ trên tập Train")
        df_movie_stats = df_train.groupby('movieId').agg(
            movie_rating_count=('rating', 'count'),
            movie_avg_rating=('rating', 'mean')
        ).reset_index()
        df_movie_stats['movie_log_count'] = np.log1p(df_movie_stats['movie_rating_count'])

        global_train_avg = float(df_train['rating'].mean())

        df_user_stats = df_train.groupby('userId').agg(
            user_rating_count=('rating', 'count'),
            user_avg_rating=('rating', 'mean')
        ).reset_index()
        df_user_stats['user_log_count'] = np.log1p(df_user_stats['user_rating_count'])

        # 1.6 Điền khuyết release_year Tầng 2 & 3
        print("1.6 Điền khuyết release_year (Tầng 2: min_timestamp, Tầng 3: median)")
        movie_min_ts = df_train.groupby('movieId')['timestamp'].min().reset_index()
        movie_min_ts['year_tier2'] = pd.to_datetime(movie_min_ts['timestamp'], unit='s').dt.year
        df_movies_final = df_movies_final.merge(movie_min_ts[['movieId', 'year_tier2']], on='movieId', how='left')

        df_movies_final['release_year'] = df_movies_final['release_year'].fillna(df_movies_final['year_tier2'])
        df_movies_final.drop(columns=['year_tier2'], inplace=True)

        median_year = df_movies_final['release_year'].median()
        df_movies_final['release_year'] = df_movies_final['release_year'].fillna(median_year).astype(int)

    return df_movies_final, df_movie_stats, df_user_stats, df_train, df_test, df_sample, global_train_avg


def train_collaborative_filtering(df_train, n_components=32):
    """
    Thực hiện Giai đoạn 2:
    - Ánh xạ ID người dùng và phim sang chỉ số ma trận liên tục.
    - Xây dựng ma trận thưa User-Item CSR Matrix.
    - Nén ma trận bằng TruncatedSVD (32 chiều tiềm ẩn).
    """
    log_step("GIAI ĐOẠN 2: Collaborative Filtering - Lọc Nhanh Ứng Viên (Candidate Retrieval Top 300)")

    unique_train_users = df_train['userId'].unique()
    unique_train_movies = df_train['movieId'].unique()

    user2idx = {u: i for i, u in enumerate(unique_train_users)}
    idx2user = {i: u for u, i in user2idx.items()}
    movie2idx = {m: i for i, m in enumerate(unique_train_movies)}
    idx2movie = {i: m for m, i in movie2idx.items()}

    print(f"Người dùng trong Train: {len(unique_train_users):,}")
    print(f"Phim trong Train: {len(unique_train_movies):,}")

    row_indices = df_train['userId'].map(user2idx).values
    col_indices = df_train['movieId'].map(movie2idx).values
    rating_values = df_train['rating'].values

    user_item_matrix = csr_matrix(
        (rating_values, (row_indices, col_indices)),
        shape=(len(unique_train_users), len(unique_train_movies)),
        dtype=np.float32
    )
    print(f'{user_item_matrix}')

    sparsity = 1.0 - (user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]))
    print(f"Ma trận thưa User-Item: {user_item_matrix.shape} (Tương tác: {user_item_matrix.nnz:,}, Độ thưa: {sparsity*100:.2f}%)")

    print(f"Huấn luyện phân rã ma trận TruncatedSVD ({n_components} chiều tiềm ẩn)")
    start_t = time.time()
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_latent = svd.fit_transform(user_item_matrix).astype(np.float32)
    movie_latent = svd.components_.T.astype(np.float32)
    print(f"Huấn luyện SVD hoàn tất trong {time.time() - start_t:.2f}s")

    return user_latent, movie_latent, user2idx, idx2user, movie2idx, idx2movie


# 2. XÂY DỰNG ĐẶC TRƯNG & LẤY MẪU ÂM LAI (PHASE 3)
def build_features(df, user_latent, movie_latent, user2idx, movie2idx,
                   df_movies_final, df_movie_stats, df_user_stats, global_train_avg):
    """
    Hàm ghép nối đặc trưng hỗn hợp cho tập tương tác (userId, movieId):
    1. Tín hiệu CF (cf_score) bằng tích vô hướng vector hóa
    2. Đặc trưng nội dung phim (release_year,thể loại One-Hot, 30 tags TF-IDF, genre_count).
    3. Tuổi phim động (movie_age) tại thời điểm tương tác (hoặc năm 2026 khi serving).
    4. Thống kê phim (movie_avg_rating, movie_log_count).
    5. Thống kê người dùng (user_avg_rating, user_log_count).
    6. Tương tác chéo (rating_diff_from_user_avg = movie_avg_rating - user_avg_rating).
    """
    df = df.copy()

    # 1 Tính toán điểm tương đồng CF (cf_score)
    u_indices = df['userId'].map(user2idx).values
    m_indices = df['movieId'].map(movie2idx).values

    valid_mask = (~pd.isna(u_indices)) & (~pd.isna(m_indices))
    cf_scores = np.zeros(len(df), dtype=np.float32)

    valid_u = u_indices[valid_mask].astype(int)
    valid_m = m_indices[valid_mask].astype(int)
    cf_scores[valid_mask] = np.sum(user_latent[valid_u] * movie_latent[valid_m], axis=1)
    df['cf_score'] = cf_scores

    # 2 Ghép đặc trưng phim từ df_movies_final
    drop_cols = ['title', 'genres', 'clean_title', 'movie_age']
    movie_feats = df_movies_final.drop(columns=[c for c in drop_cols if c in df_movies_final.columns])
    df = df.merge(movie_feats, on='movieId', how='left')

    # 3 Tính tuổi phim động (movie_age)
    if 'timestamp' in df.columns:
        interaction_year = pd.to_datetime(df['timestamp'], unit='s').dt.year
    else:
        interaction_year = 2026

    df['movie_age'] = np.maximum(0, interaction_year - df['release_year']).fillna(0).astype(int)

    # 4 Ghép thống kê phim và người dùng
    df = df.merge(df_movie_stats[['movieId', 'movie_avg_rating', 'movie_log_count']], on='movieId', how='left')
    df = df.merge(df_user_stats[['userId', 'user_avg_rating', 'user_log_count']], on='userId', how='left')

    # 5 Đặc trưng chênh lệch điểm số (rating_diff_from_user_avg)
    df['rating_diff_from_user_avg'] = df['movie_avg_rating'] - df['user_avg_rating']

    # Điền khuyết khắc phục Cold-Start
    df['movie_avg_rating'] = df['movie_avg_rating'].fillna(global_train_avg)
    df['movie_log_count'] = df['movie_log_count'].fillna(0.0)
    df['user_avg_rating'] = df['user_avg_rating'].fillna(global_train_avg)
    df['user_log_count'] = df['user_log_count'].fillna(0.0)
    df['rating_diff_from_user_avg'] = df['rating_diff_from_user_avg'].fillna(0.0)

    return df


def prepare_hybrid_training_data(df_train, df_test, df_sample, movie2idx,
                                 n_pos=250000, test_sample_size=100000):
    """
    Chiến lược Hybrid Negative Sampling:
    - 250,000 mẫu Dương (y=1): Phim user đã xem và chấm >= 4.0 sao.
    - 250,000 mẫu Âm quan sát được (y=0): Phim user đã xem nhưng chấm < 4.0 sao.
    - 250,000 mẫu Âm chưa từng xem (y=0): Bốc ngẫu nhiên từ kho phim chưa từng xem (loại trừ tập test).
    Tổng cộng: 750,000 dòng huấn luyện (hoặc tự động điều chỉnh theo kích thước train có sẵn).
    """
    print("Tiến hành tạo tập huấn luyện theo Hybrid Negative Sampling")
    user_watched_all = df_sample.groupby('userId')['movieId'].apply(set).to_dict()
    all_unique_movies = list(movie2idx.keys())

    pos_train = df_train[df_train['target'] == 1]
    neg_train_observed = df_train[df_train['target'] == 0]

    actual_n_pos = min(n_pos, len(pos_train), len(neg_train_observed))
    print(f"Lấy {actual_n_pos:,} mẫu Dương và {actual_n_pos:,} mẫu Âm quan sát được")

    sampled_pos = pos_train.sample(n=actual_n_pos, random_state=42).copy()
    sampled_neg_obs = neg_train_observed.sample(n=actual_n_pos, random_state=42).copy()

    # Tạo mẫu Âm chưa từng xem (Unobserved Negatives)
    print(f"Sinh {actual_n_pos:,} mẫu Âm ngẫu nhiên chưa từng xem")
    unobserved_neg_rows = []
    unique_sampled_users = sampled_pos['userId'].unique()
    samples_per_user = actual_n_pos // len(unique_sampled_users) + 1

    user_median_ts = df_train.groupby('userId')['timestamp'].median().to_dict()
    global_median_ts = df_train['timestamp'].median()

    random.seed(42)
    for u in unique_sampled_users:
        watched = user_watched_all.get(u, set())
        u_ts = user_median_ts.get(u, global_median_ts)
        unseen_candidates = []
        while len(unseen_candidates) < samples_per_user:
            rand_m = random.choice(all_unique_movies)
            if rand_m not in watched:
                unseen_candidates.append(rand_m)

        for m in unseen_candidates[:samples_per_user]:
            unobserved_neg_rows.append({
                'userId': u,
                'movieId': m,
                'target': 0,
                'timestamp': u_ts
            })

    df_neg_unobserved = pd.DataFrame(unobserved_neg_rows).iloc[:actual_n_pos]

    cols_to_use = ['userId', 'movieId', 'target', 'timestamp']
    train_sampled = pd.concat([
        sampled_pos[cols_to_use],
        sampled_neg_obs[cols_to_use],
        df_neg_unobserved[cols_to_use]
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    sample_test = df_test.sample(n=min(test_sample_size, len(df_test)), random_state=42).copy()

    print(f"Tập Train đã cân bằng: {len(train_sampled):,} dòng")
    print(f"Mẫu Dương (Thích >= 4.0): {len(sampled_pos):,} dòng ({len(sampled_pos)/len(train_sampled)*100:.1f}%)")
    print(f"Mẫu Âm thật (Chê < 4.0):   {len(sampled_neg_obs):,} dòng ({len(sampled_neg_obs)/len(train_sampled)*100:.1f}%)")
    print(f"Mẫu Âm chưa xem (Unseen): {len(df_neg_unobserved):,} dòng ({len(df_neg_unobserved)/len(train_sampled)*100:.1f}%)")
    print(f"Tập Test lấy mẫu: {len(sample_test):,} dòng")

    return train_sampled, sample_test


# 3. HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH LIGHTGBM (PHASE 3)
def train_lightgbm_ranker(train_features_df, test_features_df):
    """
    Huấn luyện mô hình LightGBM Classifier trên ma trận đặc trưng.
    Đánh giá độ chính xác phân hạng bằng ROC-AUC và Binary Log Loss trên tập Test.
    """
    log_step("GIAI ĐOẠN 3: Machine Learning Scoring & Re-Ranking (LightGBM Ranker)")

    exclude_cols = ['userId', 'movieId', 'rating', 'target', 'timestamp']
    feature_cols = [c for c in train_features_df.columns if c not in exclude_cols]

    print(f"Tổng số đặc trưng đưa vào LightGBM: {len(feature_cols)}")
    print(f"Kích thước X_train: {train_features_df[feature_cols].shape}")
    print(f"Kích thước X_test: {test_features_df[feature_cols].shape}")

    X_train = train_features_df[feature_cols]
    y_train = train_features_df['target']
    X_test = test_features_df[feature_cols]
    y_test = test_features_df['target']

    lgb_model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    print("Huấn luyện mô hình LightGBM Classifier")
    start_t = time.time()
    lgb_model.fit(X_train, y_train)
    print(f"Huấn luyện LightGBM hoàn tất trong {time.time() - start_t:.2f}s!")

    # Đánh giá trên tập Test
    test_preds = lgb_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds)
    test_logloss = log_loss(y_test, test_preds)

    print("-" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST:")
    print(f"ROC-AUC Score: {test_auc:.4f} (Độ chính xác phân biệt phim thích/không thích)")
    print(f"Binary Log Loss: {test_logloss:.4f} (Sai số dự đoán xác suất)")
    print("-" * 60)

    # Tầm quan trọng của các đặc trưng (Feature Importance)
    feature_importance = pd.Series(
        lgb_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    print("TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT:")
    for rank, (feat, imp) in enumerate(feature_importance.head(15).items(), 1):
        print(f"{rank:2d}. {feat}: {imp:.4f} lần phần tách ")
    return lgb_model, feature_cols, test_auc, test_logloss, feature_importance


# 4. CLASS HỆ THỐNG GỢI Ý HYBRID HOÀN CHỈNH (PHASE 4)
class HybridMovieRecommender:
    """
    Class đóng gói toàn bộ luồng gợi ý Hybrid AI 2 Vòng:
    - Vòng 1: Lọc 300 ứng viên tiềm năng bằng Collaborative Filtering (CF).
    - Vòng 2: Xếp hạng lại và dự đoán xác suất yêu thích bằng LightGBM.
    - Xử lý mượt mà cả người dùng cũ trong Train, người dùng có lịch sử ngoài Train,
      và người dùng mới hoàn toàn (Cold-Start) dựa trên Instant User Vector.
    """
    def __init__(self, user_latent, movie_latent, user2idx, movie2idx, idx2movie,
                 df_movies_final, df_movie_stats, df_user_stats, lgb_model, feature_cols,
                 user_watched_train=None, all_ratings_sample=None):
        self.user_latent = user_latent
        self.movie_latent = movie_latent
        self.user2idx = user2idx
        self.movie2idx = movie2idx
        self.idx2movie = idx2movie
        self.df_movies = df_movies_final
        self.df_movie_stats = df_movie_stats
        self.df_user_stats = df_user_stats
        self.lgb_model = lgb_model
        self.feature_cols = feature_cols
        self.user_watched = user_watched_train if user_watched_train is not None else {}
        self.all_ratings = all_ratings_sample

        self.global_user_avg_rating = float(df_user_stats['user_avg_rating'].mean())
        self.global_user_log_count = float(df_user_stats['user_log_count'].median())
        self.global_movie_avg_rating = float(df_movie_stats['movie_avg_rating'].mean())

    def search_movies(self, query, limit=5):
        """Tìm kiếm phim trong kho theo từ khóa tựa đề."""
        mask = self.df_movies['clean_title'].str.contains(query, case=False, na=False, regex=False)
        cols = ['movieId', 'clean_title', 'release_year', 'genres']
        return self.df_movies[mask][cols].head(limit)

    def _build_feature_table(self, candidate_ids, user_vector, user_avg_rating, user_log_count):
        """Tạo bảng đặc trưng phục vụ giai đoạn Serving cho các ứng viên"""
        m_indices = [self.movie2idx[m] for m in candidate_ids]
        cf_scores = user_vector.dot(self.movie_latent[m_indices].T)

        cand_df = pd.DataFrame({'movieId': candidate_ids, 'cf_score': cf_scores})

        drop_cols = ['title', 'genres', 'clean_title', 'movie_age']
        movie_feats = self.df_movies.drop(columns=[c for c in drop_cols if c in self.df_movies.columns])
        cand_df = cand_df.merge(movie_feats, on='movieId', how='left')

        # Năm 2026 làm mốc thời gian suy luận Serving
        cand_df['movie_age'] = np.maximum(0, 2026 - cand_df['release_year']).fillna(0).astype(int)
        cand_df = cand_df.merge(self.df_movie_stats[['movieId', 'movie_avg_rating', 'movie_log_count']], on='movieId', how='left')

        cand_df['user_avg_rating'] = user_avg_rating
        cand_df['user_log_count'] = user_log_count
        cand_df['rating_diff_from_user_avg'] = cand_df['movie_avg_rating'] - user_avg_rating

        cand_df['movie_avg_rating'] = cand_df['movie_avg_rating'].fillna(self.global_movie_avg_rating)
        cand_df['movie_log_count'] = cand_df['movie_log_count'].fillna(0.0)
        cand_df['rating_diff_from_user_avg'] = cand_df['rating_diff_from_user_avg'].fillna(0.0)

        return cand_df

    def _candidate_ids(self, user_vector, excluded_ids, top_n_candidates):
        """Chỉ lấy phim chưa xem, kể cả khi kho còn ít hơn 300 phim."""
        scores = user_vector.dot(self.movie_latent.T)
        available = np.ones(len(scores), dtype=bool)
        excluded_indices = [self.movie2idx[m] for m in excluded_ids if m in self.movie2idx]
        available[excluded_indices] = False
        indices = np.flatnonzero(available)
        count = min(top_n_candidates, len(indices))
        if count <= 0:
            return []
        selected = np.argpartition(scores[indices], -count)[-count:]
        indices = indices[selected]
        indices = indices[np.argsort(-scores[indices])]
        return [self.idx2movie[int(index)] for index in indices]

    def _rank_candidates(self, candidate_ids, user_vector, user_avg, user_count, top_k):
        display_cols = ['movieId', 'clean_title', 'release_year', 'genres',
                        'predicted_like_prob', 'cf_score', 'movie_avg_rating']
        if not candidate_ids:
            return pd.DataFrame(columns=display_cols)
        candidates = self._build_feature_table(candidate_ids, user_vector, user_avg, user_count)
        candidates['predicted_like_prob'] = self.lgb_model.predict_proba(candidates[self.feature_cols])[:, 1]
        ranked = candidates.sort_values('predicted_like_prob', ascending=False).head(top_k)
        metadata = self.df_movies[['movieId', 'clean_title', 'genres']].drop_duplicates('movieId')
        result = ranked.merge(metadata, on='movieId', how='left')
        return result[display_cols].reset_index(drop=True)

    def recommend_for_user(self, user_id, top_k=10, top_n_candidates=300):
        """Gợi ý phim cho người dùng đã có trong hệ thống."""
        watched_ids = set()

        if user_id in self.user2idx:
            u_idx = self.user2idx[user_id]
            user_vec = self.user_latent[u_idx]
            user_avg = self.df_user_stats.loc[self.df_user_stats['userId'] == user_id, 'user_avg_rating'].values[0]
            user_cnt = self.df_user_stats.loc[self.df_user_stats['userId'] == user_id, 'user_log_count'].values[0]
            watched_ids = self.user_watched.get(user_id, set())
        elif self.all_ratings is not None and user_id in self.all_ratings['userId'].values:
            user_hist = self.all_ratings[self.all_ratings['userId'] == user_id]
            user_avg = user_hist['rating'].mean()
            user_cnt = np.log1p(len(user_hist))
            watched_ids = set(user_hist['movieId'].values)
            liked_m = user_hist[user_hist['rating'] >= 3.5]['movieId'].values
            valid_m = [self.movie2idx[m] for m in liked_m if m in self.movie2idx]
            if len(valid_m) > 0:
                user_vec = np.mean(self.movie_latent[valid_m], axis=0)
            else:
                user_vec = np.zeros(self.movie_latent.shape[1], dtype=np.float32)
        else:
            popular_ids = self.df_movie_stats.nlargest(top_k, 'movie_rating_count')['movieId'].values
            return self.df_movies[self.df_movies['movieId'].isin(popular_ids)][['movieId', 'clean_title', 'release_year', 'genres']]

        # Vòng 1: Lọc Top 300 ứng viên qua CF
        candidate_ids = self._candidate_ids(user_vec, watched_ids, top_n_candidates)

        # Vòng 2: Xếp hạng lại bằng LightGBM
        return self._rank_candidates(candidate_ids, user_vec, user_avg, user_cnt, top_k)

    def recommend_for_new_user(self, liked_movie_ids_or_titles, top_k=10, top_n_candidates=300):
        """Giải quyết bài toán Cold-Start cho người dùng mới hoàn toàn dựa vào Instant Vector."""
        valid_movie_ids = []
        for item in liked_movie_ids_or_titles:
            if isinstance(item, int) and item in self.movie2idx:
                valid_movie_ids.append(item)
            elif isinstance(item, str):
                found = self.search_movies(item, limit=1)
                if len(found) > 0:
                    m_id = found['movieId'].iloc[0]
                    if m_id in self.movie2idx:
                        valid_movie_ids.append(m_id)

        if not valid_movie_ids:
            print("Không tìm thấy phim hợp lệ. Trả về top phim thịnh hành nhất.")
            popular_ids = self.df_movie_stats.nlargest(top_k, 'movie_rating_count')['movieId'].values
            return self.df_movies[self.df_movies['movieId'].isin(popular_ids)][['movieId', 'clean_title', 'release_year', 'genres']]

        valid_movie_ids = list(dict.fromkeys(valid_movie_ids))
        m_indices = [self.movie2idx[m] for m in valid_movie_ids]
        instant_user_vector = np.mean(self.movie_latent[m_indices], axis=0)

        # Vòng 1: Lọc 300 ứng viên qua CF
        candidate_ids = self._candidate_ids(instant_user_vector, valid_movie_ids, top_n_candidates)

        # Vòng 2: Chấm điểm xác suất qua LightGBM
        return self._rank_candidates(candidate_ids, instant_user_vector,
                                     self.global_user_avg_rating, self.global_user_log_count, top_k)


# 5. ĐÁNH GIÁ END-TO-END BENCHMARK (VÒNG 1 VS VÒNG 2)
def evaluate_end_to_end(recommender, df_test, user2idx, movie_latent, idx2movie, movie2idx,
                        user_watched_train, n_eval_users=500):
    """
    Đo lường so sánh hiệu năng gợi ý giữa Vòng 1 (Pure CF) và Vòng 2 (Hybrid LightGBM)
    trên 500 người dùng thực tế trong tập Test:
    - Precision@10: Tỷ lệ phim gợi ý đúng gu người dùng thực sự thích (rating >= 4.0).
    - HitRate@10: Tỷ lệ người dùng có ít nhất 1 phim đúng gu trong Top 10 đề xuất.
    """
    log_step(f"GIAI ĐOẠN 4: Đánh giá End-to-End Benchmark ({n_eval_users} người dùng Test)")

    test_positive = df_test[df_test['target'] == 1].groupby('userId')['movieId'].apply(set).to_dict()
    eval_users = [u for u in list(test_positive.keys()) if u in user2idx][:n_eval_users]

    cf_hits, hybrid_hits = 0, 0
    cf_precisions, hybrid_precisions = [], []

    print(f"Đang tiến hành đánh giá trên {len(eval_users)} người dùng")

    for u in eval_users:
        actual_liked = test_positive[u]
        if len(actual_liked) == 0:
            continue

        # 1 Gợi ý từ Vòng 1 (Pure CF)
        u_idx = user2idx[u]
        u_vec = recommender.user_latent[u_idx]
        scores = u_vec.dot(movie_latent.T)
        if u in user_watched_train:
            w_idx = [movie2idx[m] for m in user_watched_train[u] if m in movie2idx]
            scores[w_idx] = -1e9
        top10_idx = np.argpartition(scores, -10)[-10:]
        sorted_top10 = top10_idx[np.argsort(-scores[top10_idx])]
        cf_cands = [idx2movie[i] for i in sorted_top10]

        cf_inter = actual_liked.intersection(set(cf_cands))
        cf_precisions.append(len(cf_inter) / 10.0)
        if len(cf_inter) > 0:
            cf_hits += 1

        # 2 Gợi ý từ Vòng 2 (Hybrid LightGBM)
        hybrid_res = recommender.recommend_for_user(u, top_k=10, top_n_candidates=300)
        hybrid_cand_ids = hybrid_res['movieId'].values[:10]
        hybrid_inter = actual_liked.intersection(set(hybrid_cand_ids))
        hybrid_precisions.append(len(hybrid_inter) / 10.0)
        if len(hybrid_inter) > 0:
            hybrid_hits += 1

    cf_hr = (cf_hits / len(eval_users)) * 100
    cf_prec = np.mean(cf_precisions) * 100
    hyb_hr = (hybrid_hits / len(eval_users)) * 100
    hyb_prec = np.mean(hybrid_precisions) * 100

    print("=" * 60)
    print("KẾT QUẢ SO SÁNH HIỆU NĂNG GỢI Ý")
    print("=" * 60)
    print("1 Chỉ dùng Collaborative Filtering (Vòng 1 - Candidate Retrieval):")
    print(f"HitRate@10: {cf_hr:.2f}%")
    print(f"Precision@10: {cf_prec:.2f}%")
    print("2 Hệ thống kết hợp Hybrid CF + LightGBM Ranker (Vòng 2 - Re-Ranking):")
    print(f"HitRate@10: {hyb_hr:.2f}%")
    print(f"Precision@10: {hyb_prec:.2f}%")
    print("=" * 60)

    return {
        'cf_hitrate_10': cf_hr,
        'cf_precision_10': cf_prec,
        'hybrid_hitrate_10': hyb_hr,
        'hybrid_precision_10': hyb_prec
    }


# 6. LƯU TRỮ ARTIFACTS MÔ HÌNH CHO GIAI ĐOẠN 4
def save_model_artifacts(data_dir, lgb_model, feature_cols, metrics):
    """Lưu trữ mô hình LightGBM booster và danh sách đặc trưng ra đĩa."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    model_path = data_dir / "lgbm_ranker.txt"
    lgb_model.booster_.save_model(str(model_path))

    cols_path = data_dir / "feature_cols.pkl"
    with open(cols_path, "wb") as f:
        pickle.dump(feature_cols, f)

    metrics_path = data_dir / "phase3_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"Đã lưu thành công các Artifacts của Phase 3 tại {data_dir}:")


def save_serving_artifacts(data_dir, recommender, df_train):
    """Lưu đúng đầu vào suy luận của lần huấn luyện, kèm lịch sử chỉ từ train."""
    data_dir = Path(data_dir)
    model_path = data_dir / 'lgbm_ranker.txt'
    if not model_path.is_file():
        raise FileNotFoundError('Hãy lưu mô hình trước khi lưu bộ dữ liệu suy luận.')
    users = df_train['userId'].map(recommender.user2idx)
    movies = df_train['movieId'].map(recommender.movie2idx)
    if users.isna().any() or movies.isna().any():
        raise ValueError('Lịch sử train không khớp bộ ánh xạ CF.')
    history = csr_matrix(
        (df_train['rating'].to_numpy(dtype=np.float32),
         (users.to_numpy(dtype=np.int32), movies.to_numpy(dtype=np.int32))),
        shape=(len(recommender.user_latent), len(recommender.movie_latent)),
    )
    bundle = {
        'version': 1,
        'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'feature_cols': recommender.feature_cols,
        'user_latent': recommender.user_latent,
        'movie_latent': recommender.movie_latent,
        'user2idx': recommender.user2idx,
        'movie2idx': recommender.movie2idx,
        'idx2movie': recommender.idx2movie,
        'df_movies_final': recommender.df_movies,
        'df_movie_stats': recommender.df_movie_stats,
        'df_user_stats': recommender.df_user_stats,
        'history': history,
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=data_dir, prefix='serving-', suffix='.tmp', delete=False) as stream:
            temporary_path = Path(stream.name)
            pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, data_dir / 'serving_artifacts.pkl')
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(f"Đã lưu bộ dữ liệu suy luận: {data_dir / 'serving_artifacts.pkl'}")


# 7 HÀM MAIN THỰC THI TOÀN BỘ PIPELINE
def main():
    parser = argparse.ArgumentParser(description="MovieLens 32M Hybrid AI Recommender System")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Thư mục chứa dữ liệu và mô hình")
    parser.add_argument("--zip-path", type=str, default=None, help="Đường dẫn tệp ml-32m.zip")
    parser.add_argument("--users", type=int, default=20000, help="Số lượng người dùng tích cực nhất (mặc định 20,000)")
    parser.add_argument("--train-samples", type=int, default=250000, help="Số mẫu dương cho tập Train (mặc định 250,000)")
    parser.add_argument("--eval-users", type=int, default=500, help="Số người dùng kiểm định Benchmark (mặc định 500)")
    parser.add_argument("--save-models", action="store_true", default=True, help="Lưu trữ mô hình và bộ đặc trưng")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    zip_path = Path(args.zip_path) if args.zip_path else check_and_download_dataset(data_dir)

    # Giai đoạn 1
    (df_movies_final, df_movie_stats, df_user_stats,
     df_train, df_test, df_sample, global_train_avg) = process_phase1_data(zip_path, n_users=args.users)

    # Giai đoạn 2
    (user_latent, movie_latent,
     user2idx, idx2user, movie2idx, idx2movie) = train_collaborative_filtering(df_train, n_components=32)

    user_watched_train = df_train.groupby('userId')['movieId'].apply(set).to_dict()

    # Giai đoạn 3: Feature Engineering & Sampling
    train_sampled, sample_test = prepare_hybrid_training_data(
        df_train=df_train,
        df_test=df_test,
        df_sample=df_sample,
        movie2idx=movie2idx,
        n_pos=args.train_samples,
        test_sample_size=100000
    )

    print("Xây dựng ma trận đặc trưng cho tập train")
    train_features_df = build_features(
        train_sampled, user_latent, movie_latent, user2idx, movie2idx,
        df_movies_final, df_movie_stats, df_user_stats, global_train_avg
    )

    print("Xây dựng ma trận đặc trưng cho tập Test")
    test_features_df = build_features(
        sample_test, user_latent, movie_latent, user2idx, movie2idx,
        df_movies_final, df_movie_stats, df_user_stats, global_train_avg
    )

    # Huấn luyện LightGBM
    lgb_model, feature_cols, test_auc, test_logloss, feat_imp = train_lightgbm_ranker(
        train_features_df, test_features_df
    )

    # Khởi tạo đối tượng HybridMovieRecommender
    recommender = HybridMovieRecommender(
        user_latent=user_latent,
        movie_latent=movie_latent,
        user2idx=user2idx,
        movie2idx=movie2idx,
        idx2movie=idx2movie,
        df_movies_final=df_movies_final,
        df_movie_stats=df_movie_stats,
        df_user_stats=df_user_stats,
        lgb_model=lgb_model,
        feature_cols=feature_cols,
        user_watched_train=user_watched_train,
        all_ratings_sample=df_sample
    )

    # Thử nghiệm gợi ý cho User ID = 10
    sample_user_id = 10
    print(f"KẾT QUẢ GỢI Ý TOP 10 PHIM CHO USER ID = {sample_user_id}:")
    rec_u10 = recommender.recommend_for_user(sample_user_id, top_k=10)
    print(rec_u10.to_string(index=False))

    # Thử nghiệm Cold-Start cho người dùng mới
    print("KẾT QUẢ THỬ NGHIỆM COLD-START CHO NGƯỜI DÙNG MỚI (Thích Action/Sci-Fi):")
    new_user_picks = ['Inception', 'The Matrix', 'Fight Club']
    rec_new = recommender.recommend_for_new_user(new_user_picks, top_k=10)
    print(rec_new.to_string(index=False))

    # Đánh giá End-to-End Benchmark
    benchmark_metrics = evaluate_end_to_end(
        recommender=recommender,
        df_test=df_test,
        user2idx=user2idx,
        movie_latent=movie_latent,
        idx2movie=idx2movie,
        movie2idx=movie2idx,
        user_watched_train=user_watched_train,
        n_eval_users=args.eval_users
    )

    # Lưu trữ mô hình nếu được yêu cầu
    if args.save_models:
        all_metrics = {
            'test_roc_auc': float(test_auc),
            'test_log_loss': float(test_logloss),
            'benchmark': benchmark_metrics
        }
        save_model_artifacts(data_dir, lgb_model, feature_cols, all_metrics)
        save_serving_artifacts(data_dir, recommender, df_train)



if __name__ == "__main__":
    main()
