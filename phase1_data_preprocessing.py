import os
import re
import sys
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Thiết lập đường dẫn thư mục làm việc và dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
ZIP_PATH = os.path.join(DATA_DIR, "ml-32m.zip")

os.makedirs(DATA_DIR, exist_ok=True)


def check_and_download_data():
    """Kiểm tra và tải dữ liệu nếu chưa có sẵn."""
    if not os.path.exists(ZIP_PATH):
        print("Tệp dữ liệu ml-32m.zip chưa tồn tại. Đang tải từ GroupLens (238 MB)...")
        url = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
        urllib.request.urlretrieve(url, ZIP_PATH)
        print("Tải dữ liệu hoàn tất!")
    else:
        print(f"Đã tìm thấy tệp dữ liệu: {ZIP_PATH}")


def process_movies_and_tags():
    """Bước 1.1, 1.2 & 1.3: Xử lý phim, thể loại và trích xuất TF-IDF từ tags."""
    print("\n" + "="*60)
    print("BƯỚC 1.1, 1.2 & 1.3: XỬ LÝ PHIM, THỂ LOẠI & TAGS TF-IDF")
    print("="*60)

    # 1. Đọc movies.csv
    print("1. Đang đọc movies.csv từ zip...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        with z.open("ml-32m/movies.csv") as f:
            df_movies = pd.read_csv(f)

    print(f"   -> Tổng số phim ban đầu: {len(df_movies):,}")

    # Trích xuất năm phát hành bằng Regex
    def extract_year(title):
        match = re.search(r'\((\d{4})\)', str(title))
        return int(match.group(1)) if match else np.nan

    df_movies['release_year'] = df_movies['title'].apply(extract_year)
    median_year = df_movies['release_year'].median()
    df_movies['release_year'] = df_movies['release_year'].fillna(median_year).astype(int)
    df_movies['movie_age'] = 2026 - df_movies['release_year']
    df_movies['clean_title'] = df_movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()

    # Mã hóa Multi-hot 18 thể loại
    all_genres = [
        'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
        'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    for g in all_genres:
        pattern = r"Children" if g == "Children" else g
        df_movies[f'genre_{g}'] = df_movies['genres'].str.contains(pattern, regex=True).astype(int)

    df_movies['genre_count'] = df_movies[[f'genre_{g}' for g in all_genres]].sum(axis=1)

    # 2. Xử lý tags.csv
    print("2. Đang đọc và tổng hợp tags.csv bằng TF-IDF...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        with z.open("ml-32m/tags.csv") as f:
            df_tags = pd.read_csv(f, usecols=['movieId', 'tag'])

    df_tags = df_tags.dropna(subset=['tag'])
    df_tags['tag'] = df_tags['tag'].astype(str).str.lower().str.strip()
    print(f"   -> Tổng số lượt tag hợp lệ: {len(df_tags):,}")

    movie_tags = df_tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.astype(str))).reset_index()

    # Trích xuất 30 đặc trưng tag nổi bật nhất bằng TF-IDF
    tfidf = TfidfVectorizer(max_features=30, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movie_tags['tag'])
    tag_feature_names = [f'tag_{re.sub(r"[^a-zA-Z0-9_]", "_", feat)}' for feat in tfidf.get_feature_names_out()]

    df_tag_features = pd.DataFrame(tfidf_matrix.toarray(), columns=tag_feature_names)
    df_tag_features['movieId'] = movie_tags['movieId']

    # Ghép các đặc trưng phim
    df_movies_final = df_movies.merge(df_tag_features, on='movieId', how='left')
    for col in tag_feature_names:
        df_movies_final[col] = df_movies_final[col].fillna(0.0)

    output_path = os.path.join(DATA_DIR, "movies_processed.csv.gz")
    df_movies_final.to_csv(output_path, index=False, compression='gzip')
    print(f"Đã lưu: {output_path} (Kích thước: {df_movies_final.shape})")
    return df_movies_final


def process_ratings_and_split():
    """Bước 1.4 & 1.5: Quét 32M ratings, tính stats, gán nhãn và chia tập Train/Test."""
    print("\n" + "="*60)
    print("BƯỚC 1.4 & 1.5: THỐNG KÊ TOÀN CỤC RATINGS & TẠO TẬP TRAIN/TEST")
    print("="*60)

    chunk_size = 2_000_000
    movie_sum = {}
    movie_count = {}
    user_sum = {}
    user_count = {}
    total_rows = 0

    print("1. Đang quét 32,000,204 ratings bằng streaming chunking...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        with z.open("ml-32m/ratings.csv") as f:
            reader = pd.read_csv(f, chunksize=chunk_size, dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32, 'timestamp': np.int64})
            for chunk in reader:
                total_rows += len(chunk)
                # Movie stats
                m_grp = chunk.groupby('movieId')['rating'].agg(['sum', 'count'])
                for m_id, row in m_grp.iterrows():
                    movie_sum[m_id] = movie_sum.get(m_id, 0.0) + row['sum']
                    movie_count[m_id] = movie_count.get(m_id, 0) + int(row['count'])
                # User stats
                u_grp = chunk.groupby('userId')['rating'].agg(['sum', 'count'])
                for u_id, row in u_grp.iterrows():
                    user_sum[u_id] = user_sum.get(u_id, 0.0) + row['sum']
                    user_count[u_id] = user_count.get(u_id, 0) + int(row['count'])
                print(f"   -> Đã quét {total_rows:,} dòng...")

    print(f"Hoàn tất quét toàn bộ {total_rows:,} dòng ratings!")

    # Bảng Movie Stats
    df_movie_stats = pd.DataFrame({
        'movieId': list(movie_count.keys()),
        'movie_rating_count': list(movie_count.values()),
        'movie_avg_rating': [movie_sum[m] / movie_count[m] for m in movie_count.keys()]
    })
    df_movie_stats['movie_log_count'] = np.log1p(df_movie_stats['movie_rating_count'])

    # Bảng User Stats
    df_user_stats = pd.DataFrame({
        'userId': list(user_count.keys()),
        'user_rating_count': list(user_count.values()),
        'user_avg_rating': [user_sum[u] / user_count[u] for u in user_count.keys()]
    })
    df_user_stats['user_log_count'] = np.log1p(df_user_stats['user_rating_count'])

    df_movie_stats.to_csv(os.path.join(DATA_DIR, "movie_stats.csv.gz"), index=False, compression='gzip')
    df_user_stats.to_csv(os.path.join(DATA_DIR, "user_stats.csv.gz"), index=False, compression='gzip')
    print("Đã lưu movie_stats.csv.gz và user_stats.csv.gz!")

    # Tạo tập tương tác đại diện (20,000 users tích cực nhất)
    print("\n2. Lấy mẫu 20,000 người dùng tích cực và chia Temporal Split...")
    top_users = df_user_stats[df_user_stats['user_rating_count'] >= 30].nlargest(20000, 'user_rating_count')['userId'].values
    top_users_set = set(top_users)
    print(f"   -> Đã chọn {len(top_users_set):,} users tích cực.")

    sampled_chunks = []
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        with z.open("ml-32m/ratings.csv") as f:
            reader = pd.read_csv(f, chunksize=chunk_size, dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32, 'timestamp': np.int64})
            for chunk in reader:
                chunk_filtered = chunk[chunk['userId'].isin(top_users_set)]
                if len(chunk_filtered) > 0:
                    sampled_chunks.append(chunk_filtered)

    df_sample = pd.concat(sampled_chunks, ignore_index=True)
    df_sample['target'] = (df_sample['rating'] >= 4.0).astype(np.int8)
    df_sample = df_sample.sort_values(['userId', 'timestamp']).reset_index(drop=True)

    # Temporal split: 80% trước là Train, 20% gần đây là Test
    df_sample['rank_pct'] = df_sample.groupby('userId').cumcount() / df_sample.groupby('userId')['timestamp'].transform('count')
    train_mask = df_sample['rank_pct'] < 0.8
    df_train = df_sample[train_mask].drop(columns=['rank_pct'])
    df_test = df_sample[~train_mask].drop(columns=['rank_pct'])

    train_path = os.path.join(DATA_DIR, "train_interactions.csv.gz")
    test_path = os.path.join(DATA_DIR, "test_interactions.csv.gz")
    df_train.to_csv(train_path, index=False, compression='gzip')
    df_test.to_csv(test_path, index=False, compression='gzip')

    print(f"Tập Train: {len(df_train):,} dòng ({df_train['target'].mean()*100:.1f}% nhãn 1) -> Đã lưu: {train_path}")
    print(f"Tập Test:  {len(df_test):,} dòng ({df_test['target'].mean()*100:.1f}% nhãn 1) -> Đã lưu: {test_path}")


def main():
    print("="*60)
    print("KHỞI ĐỘNG PIPELINE GIAI ĐOẠN 1 (PHASE 1)")
    print("="*60)
    check_and_download_data()
    process_movies_and_tags()
    process_ratings_and_split()
    print("\HOÀN TẤT TOÀN BỘ GIAI ĐOẠN 1 THÀNH CÔNG!")


if __name__ == "__main__":
    main()
