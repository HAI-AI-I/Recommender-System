"""
Hệ Thống Gợi Ý Phim Hybrid AI (MovieLens 32M)
=====================================================
GIAI ĐOẠN 2: Collaborative Filtering - Lọc Nhanh Ứng Viên (Candidate Retrieval Top 300)
Được đồng bộ từ Google Colab Notebook: MovieLens_32M_Hybrid_RecSys.ipynb

Bao gồm:
- Bước 2.1: Xây dựng bộ từ điển ánh xạ (user2idx, movie2idx)
- Bước 2.2: Khởi tạo ma trận thưa User-Item (scipy.sparse.csr_matrix)
- Bước 2.3: Huấn luyện phân rã ma trận TruncatedSVD (32 latent dimensions)
- Bước 2.4: Hàm trích xuất nhanh Top 300 ứng viên (Fast Dot Product Retrieval)
- Bước 2.5: Thử nghiệm gợi ý ứng viên cho người dùng mẫu
- Bước 2.6: Đánh giá chất lượng lọc ứng viên trên tập Test (Hit Rate@300 & Recall@300)
- Bước 2.7: Lưu trữ ma trận tiềm ẩn và bộ ánh xạ để phục vụ Giai đoạn 3 (Ranking)
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Thiết lập đường dẫn thư mục làm việc và dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")

train_path = os.path.join(DATA_DIR, "train_interactions.csv.gz")
test_path = os.path.join(DATA_DIR, "test_interactions.csv.gz")
movies_path = os.path.join(DATA_DIR, "movies_processed.csv.gz")
movie_stats_path = os.path.join(DATA_DIR, "movie_stats.csv.gz")


def main():
    print("="*60)
    print("KHỞI ĐỘNG GIAI ĐOẠN 2: COLLABORATIVE FILTERING RETRIEVAL")
    print("="*60)

    # 1. Đọc dữ liệu
    print("1. Đang tải tập dữ liệu tương tác huấn luyện và đặc trưng...")
    t0 = time.time()
    df_train = pd.read_csv(train_path, usecols=['userId', 'movieId', 'rating'])
    df_test = pd.read_csv(test_path, usecols=['userId', 'movieId', 'target'])
    df_movies = pd.read_csv(movies_path, usecols=['movieId', 'clean_title', 'release_year', 'genres'])
    df_movie_stats = pd.read_csv(movie_stats_path, usecols=['movieId', 'movie_rating_count'])
    print(f"✅ Đã tải dữ liệu trong {time.time() - t0:.2f}s!")

    # 2. Xây dựng bộ ánh xạ
    print("\n2. Xây dựng bộ từ điển ánh xạ (Index Mapping)...")
    unique_train_users = df_train['userId'].unique()
    unique_train_movies = df_train['movieId'].unique()
    print(f"   -> Số lượng Users trong tập Train: {len(unique_train_users):,}")
    print(f"   -> Số lượng Movies trong tập Train: {len(unique_train_movies):,}")

    user2idx = {u: i for i, u in enumerate(unique_train_users)}
    idx2user = {i: u for u, i in user2idx.items()}
    movie2idx = {m: i for i, m in enumerate(unique_train_movies)}
    idx2movie = {i: m for m, i in movie2idx.items()}

    # 3. Tạo ma trận thưa User-Item
    print("\n3. Khởi tạo ma trận thưa User-Item CSR Matrix...")
    row_indices = df_train['userId'].map(user2idx).values
    col_indices = df_train['movieId'].map(movie2idx).values
    rating_values = df_train['rating'].values

    user_item_matrix = csr_matrix(
        (rating_values, (row_indices, col_indices)),
        shape=(len(unique_train_users), len(unique_train_movies)),
        dtype=np.float32
    )
    sparsity = 1.0 - (user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]))
    print(f"✅ Kích thước ma trận: {user_item_matrix.shape}")
    print(f"✅ Số tương tác thực tế (Non-zeros): {user_item_matrix.nnz:,}")
    print(f"✅ Độ thưa thớt (Sparsity): {sparsity * 100:.2f}%")

    # 4. Huấn luyện TruncatedSVD
    print("\n4. Đang phân rã ma trận bằng TruncatedSVD (32 latent dimensions)...")
    t0 = time.time()
    svd = TruncatedSVD(n_components=32, random_state=42)
    user_latent = svd.fit_transform(user_item_matrix)
    movie_latent = svd.components_.T
    print(f"✅ Huấn luyện hoàn tất trong {time.time() - t0:.2f}s!")
    print(f"   -> User Latent Matrix: {user_latent.shape}")
    print(f"   -> Movie Latent Matrix: {movie_latent.shape}")

    # 5. Xây dựng hàm lấy nhanh Top 300 ứng viên
    user_watched_train = df_train.groupby('userId')['movieId'].apply(set).to_dict()

    def get_top_candidates(user_id, top_n=300, exclude_watched=True):
        if user_id not in user2idx:
            top_popular = df_movie_stats.nlargest(top_n, 'movie_rating_count')['movieId'].values
            return top_popular, np.ones(top_n, dtype=np.float32)

        u_idx = user2idx[user_id]
        user_vector = user_latent[u_idx]
        scores = user_vector.dot(movie_latent.T)

        if exclude_watched and user_id in user_watched_train:
            watched_movie_ids = user_watched_train[user_id]
            watched_indices = [movie2idx[m] for m in watched_movie_ids if m in movie2idx]
            scores[watched_indices] = -1e9

        top_indices = np.argpartition(scores, -top_n)[-top_n:]
        sorted_top_indices = top_indices[np.argsort(-scores[top_indices])]

        candidate_movie_ids = [idx2movie[i] for i in sorted_top_indices]
        candidate_cf_scores = scores[sorted_top_indices]
        return candidate_movie_ids, candidate_cf_scores

    # 6. Thử nghiệm trên một user mẫu
    sample_user_id = unique_train_users[0]
    print(f"\n5. Thử nghiệm gợi ý cho User ID {sample_user_id}...")
    candidates, cf_scores = get_top_candidates(sample_user_id, top_n=10)
    df_preview = df_movies[df_movies['movieId'].isin(candidates)].copy()
    score_map = dict(zip(candidates, cf_scores))
    df_preview['cf_score'] = df_preview['movieId'].map(score_map)
    df_preview = df_preview.sort_values('cf_score', ascending=False).reset_index(drop=True)
    print("Top 10 phim ứng viên được Collaborative Filtering gợi ý:")
    print(df_preview[['movieId', 'clean_title', 'release_year', 'genres', 'cf_score']])

    # 7. Đánh giá trên tập Test
    print("\n6. Đang đánh giá trên 500 người dùng trong tập Test...")
    test_positive = df_test[df_test['target'] == 1].groupby('userId')['movieId'].apply(set).to_dict()
    eval_users = [u for u in list(test_positive.keys()) if u in user2idx][:500]

    hits = 0
    recalls = []
    for u in eval_users:
        actual_liked_movies = test_positive[u]
        if len(actual_liked_movies) == 0:
            continue
        pred_candidates, _ = get_top_candidates(u, top_n=300)
        intersection = actual_liked_movies.intersection(set(pred_candidates))
        if len(intersection) > 0:
            hits += 1
        recalls.append(len(intersection) / len(actual_liked_movies))

    print(f"📊 Số lượng người dùng kiểm thử: {len(eval_users)}")
    print(f"🎯 Hit Rate@300: {hits / len(eval_users) * 100:.2f}% (Tỷ lệ gợi ý trúng ít nhất 1 phim đúng gu)")
    print(f"🎯 Mean Recall@300: {np.mean(recalls) * 100:.2f}% (Độ bao phủ các phim yêu thích)")

    # 8. Lưu trữ embeddings & mappings
    print("\n7. Đang lưu trữ embeddings và mappings...")
    np.save(os.path.join(DATA_DIR, 'user_latent.npy'), user_latent)
    np.save(os.path.join(DATA_DIR, 'movie_latent.npy'), movie_latent)
    mappings = {
        'user2idx': user2idx,
        'idx2user': idx2user,
        'movie2idx': movie2idx,
        'idx2movie': idx2movie
    }
    with open(os.path.join(DATA_DIR, 'cf_mappings.pkl'), 'wb') as f:
        pickle.dump(mappings, f)

    print("✅ Đã lưu: user_latent.npy, movie_latent.npy, cf_mappings.pkl")
    print("\n🎉 HOÀN THÀNH TOÀN BỘ GIAI ĐOẠN 2!")


if __name__ == "__main__":
    main()
