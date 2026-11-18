"""
Hệ Thống Gợi Ý Phim Hybrid AI (MovieLens 32M)
=====================================================
GIAI ĐOẠN 2 - CÁCH 3: Content-Based Filtering (CBF) - Candidate Retrieval Top 300

Các đặc trưng kỹ thuật:
- Phân tích nội dung phim đa chiều (Multi-attribute Content Representation):
  + 18 đặc trưng thể loại Multi-hot (Genre Features).
  + 30 đặc trưng từ khóa chủ đề TF-IDF (TF-IDF Tag Features).
  + Đặc trưng năm phát hành chuẩn hóa MinMax (Release Year Feature).
- Ma trận đặc trưng phim chuẩn hóa L2-norm (Movie Feature Matrix: 87,585 x 49).
- Học hồ sơ sở thích người dùng (User Preference Profile) theo thuật toán Rocchio:
  + Tích hợp phim yêu thích (Positive Feedback: rating >= 4.0, target = 1).
  + Tích hợp phim không thích (Negative Feedback: rating <= 2.5) với hệ số phạt lambda.
  + Chuẩn hóa L2-norm cho vector User Profile.
- Truy xuất cực nhanh bằng Cosine Similarity vector hóa: CosSim(u, i) = u_profile . v_movie.
- Loại trừ các phim người dùng đã tương tác trong tập Train.
- Cơ chế Fallback tách biệt cho người dùng mới (Cold-Start Users).
- Lưu trữ và nạp artifacts độc lập trong dataset/content_based/.
- Chuẩn hóa đầu ra chuyển giao cho Phase 3 (LightGBM Ranker):
  ['userId', 'movieId', 'candidate_score', 'candidate_source']
"""

import os
import sys
import time
import json
import pickle
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any

import numpy as np
import pandas as pd
from scipy.sparse import save_npz, load_npz, csr_matrix
from sklearn.preprocessing import normalize
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ContentBasedFiltering")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = DATA_DIR / "content_based"


@dataclass
class ContentBasedConfig:
    """Cấu hình các siêu tham số cho Content-Based Filtering."""
    w_genre: float = 0.45              
    w_tag: float = 0.45               
    w_year: float = 0.10               
    neg_penalty_lambda: float = 0.10   
    min_positive_rating: float = 4.0 
    max_negative_rating: float = 2.5  
    top_n_candidates: int = 300      
    random_seed: int = 42           

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentBasedConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ContentBasedRecommender:
    """Hệ thống Candidate Retrieval dựa trên Content-Based Filtering thuần túy."""

    def __init__(self, config: Optional[ContentBasedConfig] = None):
        self.config = config or ContentBasedConfig()
        self.movie2idx: Dict[int, int] = {}
        self.idx2movie: Dict[int, int] = {}
        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.movie_feature_matrix: Optional[np.ndarray] = None  # (n_movies, n_features)
        self.user_profiles: Optional[np.ndarray] = None          # (n_users, n_features)
        self.user_watched_train: Dict[int, Set[int]] = {}
        self.popular_fallback_movies: List[int] = []
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        self.fit_metrics: Dict[str, Any] = {}

    def _get_peak_ram_mb(self) -> float:
        """Đo lường RAM đang sử dụng của tiến trình (MB)."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def build_movie_features(self, df_movies: pd.DataFrame) -> np.ndarray:
        """
        Xây dựng Ma trận đặc trưng phim (Movie Feature Matrix):
        1. Trích xuất và chuẩn hóa 18 đặc trưng thể loại (Multi-hot).
        2. Trích xuất và chuẩn hóa 30 đặc trưng từ khóa TF-IDF.
        3. Chuẩn hóa đặc trưng năm phát hành MinMax [0, 1].
        4. Kết hợp có trọng số (Weighted combination) và chuẩn hóa L2-norm toàn vector.
        """
        logger.info("Đang xây dựng Movie Feature Matrix từ metadata phim...")
        t0 = time.time()

        unique_movies = df_movies["movieId"].values
        self.movie2idx = {int(m): int(i) for i, m in enumerate(unique_movies)}
        self.idx2movie = {int(i): int(m) for m, i in self.movie2idx.items()}

        genre_cols = [c for c in df_movies.columns if c.startswith("genre_") and c != "genre_count"]
        tag_cols = [c for c in df_movies.columns if c.startswith("tag_")]
        self.feature_names = genre_cols + tag_cols + ["release_year_norm"]

        logger.info(f"Tổng số phim: {len(df_movies):,}")
        logger.info(f"Nhóm thể loại: {len(genre_cols)} cột | Nhóm Tags TF-IDF: {len(tag_cols)} cột | Năm: 1 cột")
        G = df_movies[genre_cols].values.astype(np.float32)
        G_norm = normalize(G, norm="l2", axis=1)
        T = df_movies[tag_cols].values.astype(np.float32)
        T_norm = normalize(T, norm="l2", axis=1)
        y_min = df_movies["release_year"].min()
        y_max = df_movies["release_year"].max()
        if y_max > y_min:
            Y_norm = ((df_movies["release_year"] - y_min) / (y_max - y_min)).values.reshape(-1, 1).astype(np.float32)
        else:
            Y_norm = np.zeros((len(df_movies), 1), dtype=np.float32)
        w_g = self.config.w_genre
        w_t = self.config.w_tag
        w_y = self.config.w_year

        M_combined = np.hstack([
            w_g * G_norm,
            w_t * T_norm,
            w_y * Y_norm
        ])
        self.movie_feature_matrix = normalize(M_combined, norm="l2", axis=1).astype(np.float32)
        dur = time.time() - t0
        mem_mb = self.movie_feature_matrix.nbytes / (1024 * 1024)
        logger.info(f"Đã xây dựng xong Movie Feature Matrix: {self.movie_feature_matrix.shape} ({mem_mb:.2f} MB) trong {dur:.2f}s!")
        return self.movie_feature_matrix

    def fit(
        self,
        df_train: pd.DataFrame,
        df_movies: pd.DataFrame,
        df_movie_stats: Optional[pd.DataFrame] = None
    ) -> "ContentBasedRecommender":
        """
        Huấn luyện Content-Based Filtering:
        1. Xây dựng Movie Feature Matrix.
        2. Học hồ sơ người dùng (User Profile) từ lịch sử tương tác trong Train.
        3. Chuẩn bị danh sách fallback cho người dùng cold-start.
        """
        logger.info("=" * 60)
        logger.info("BẮT ĐẦU HUẤN LUYỆN CONTENT-BASED FILTERING (CBF)")
        logger.info("=" * 60)
        t_start = time.time()
        ram_start = self._get_peak_ram_mb()
        self.build_movie_features(df_movies)
        unique_users = df_train["userId"].unique()
        self.user2idx = {int(u): int(i) for i, u in enumerate(unique_users)}
        self.idx2user = {int(i): int(u) for u, i in self.user2idx.items()}
        n_users = len(unique_users)
        n_features = self.movie_feature_matrix.shape[1]

        logger.info(f"Số lượng Users trong tập Train: {n_users:,}")
        logger.info("Đang lập danh mục phim đã xem trong Train cho từng người dùng...")
        self.user_watched_train = df_train.groupby("userId")["movieId"].apply(set).to_dict()

        if df_movie_stats is not None and "movie_rating_count" in df_movie_stats.columns:
            sorted_stats = df_movie_stats.sort_values("movie_rating_count", ascending=False)
            self.popular_fallback_movies = sorted_stats["movieId"].head(self.config.top_n_candidates).tolist()
        else:
            top_movies = df_train["movieId"].value_counts().head(self.config.top_n_candidates).index.tolist()
            self.popular_fallback_movies = [int(m) for m in top_movies]
        logger.info("Đang xây dựng User Profiles (Rocchio Preference Learning)...")
        user_profiles_mat = np.zeros((n_users, n_features), dtype=np.float32)

        train_grouped = df_train.groupby("userId")
        t_prof_start = time.time()

        lamb = self.config.neg_penalty_lambda
        pos_min = self.config.min_positive_rating
        neg_max = self.config.max_negative_rating

        for u, group in train_grouped:
            u_idx = self.user2idx[u]

            # Phim yêu thích (Positive Interactions)
            pos_mask = (group["rating"] >= pos_min) | (group["target"] == 1)
            pos_df = group[pos_mask]

            # Nếu không có phim >= 4.0, lấy top các phim có điểm cao nhất của user
            if len(pos_df) == 0:
                pos_df = group.nlargest(10, "rating")

            pos_movie_idx = [self.movie2idx[m] for m in pos_df["movieId"] if m in self.movie2idx]
            if len(pos_movie_idx) > 0:
                # Trọng số theo độ chênh lệch rating: w = rating - 3.0
                pos_weights = (pos_df["rating"] - 3.0).values.reshape(-1, 1).astype(np.float32)
                pos_profile = np.sum(pos_weights * self.movie_feature_matrix[pos_movie_idx], axis=0)
            else:
                pos_profile = np.zeros(n_features, dtype=np.float32)

            # Phim không thích (Negative Feedback)
            neg_df = group[group["rating"] <= neg_max]
            if len(neg_df) > 0 and lamb > 0:
                neg_movie_idx = [self.movie2idx[m] for m in neg_df["movieId"] if m in self.movie2idx]
                if len(neg_movie_idx) > 0:
                    neg_weights = (3.0 - neg_df["rating"]).values.reshape(-1, 1).astype(np.float32)
                    neg_profile = np.sum(neg_weights * self.movie_feature_matrix[neg_movie_idx], axis=0)
                    user_vec = pos_profile - (lamb * neg_profile)
                else:
                    user_vec = pos_profile
            else:
                user_vec = pos_profile

            # Chuẩn hóa L2-norm cho User Profile
            norm = np.linalg.norm(user_vec)
            if norm > 1e-7:
                user_profiles_mat[u_idx] = user_vec / norm
            else:
                user_profiles_mat[u_idx] = user_vec

        self.user_profiles = user_profiles_mat
        t_prof_dur = time.time() - t_prof_start
        total_fit_dur = time.time() - t_start
        ram_end = self._get_peak_ram_mb()
        ram_delta = max(0.0, ram_end - ram_start)

        self.fit_metrics = {
            "num_movies": len(self.movie2idx),
            "num_users": n_users,
            "feature_dim": n_features,
            "feature_building_time_sec": round(time.time() - t_start - t_prof_dur, 2),
            "user_profile_time_sec": round(t_prof_dur, 2),
            "total_fit_time_sec": round(total_fit_dur, 2),
            "user_profiles_memory_mb": round(self.user_profiles.nbytes / (1024 * 1024), 2),
            "movie_features_memory_mb": round(self.movie_feature_matrix.nbytes / (1024 * 1024), 2),
            "peak_ram_mb": round(ram_end, 2)
        }

        self.is_fitted = True
        logger.info(f"Xây dựng User Profiles hoàn tất trong {t_prof_dur:.2f}s!")
        logger.info(f"Tổng thời gian huấn luyện: {total_fit_dur:.2f}s | RAM đỉnh: {ram_end:.1f} MB")
        return self

    def get_top_candidates(
        self,
        user_id: int,
        top_n: int = 300,
        exclude_watched: bool = True
    ) -> Tuple[List[int], np.ndarray, bool]:
        """
        Trích xuất Top-N phim ứng viên cho người dùng bằng Cosine Similarity:
        Score(u, i) = UserProfile(u) . MovieVector(i)
        Loại trừ các phim người dùng đã xem trong Train.
        """
        if not self.is_fitted or self.movie_feature_matrix is None:
            raise RuntimeError("Mô hình chưa được fit hoặc chưa nạp artifacts.")
        if user_id not in self.user2idx or self.user_profiles is None:
            fallback = self.popular_fallback_movies[:top_n]
            return fallback, np.ones(len(fallback), dtype=np.float32), True

        u_idx = self.user2idx[user_id]
        u_profile = self.user_profiles[u_idx]
        if np.all(u_profile == 0.0):
            fallback = self.popular_fallback_movies[:top_n]
            return fallback, np.ones(len(fallback), dtype=np.float32), True

        scores = u_profile.dot(self.movie_feature_matrix.T)
        if exclude_watched and user_id in self.user_watched_train:
            watched_ids = self.user_watched_train[user_id]
            watched_indices = [self.movie2idx[m] for m in watched_ids if m in self.movie2idx]
            scores[watched_indices] = -1e9
        valid_indices = np.where(scores > -1e8)[0]
        n_retrieve = min(top_n, len(valid_indices))

        if n_retrieve == 0:
            watched_set = self.user_watched_train.get(user_id, set())
            unwatched_popular = [m for m in self.popular_fallback_movies if m not in watched_set][:top_n]
            return unwatched_popular, np.ones(len(unwatched_popular), dtype=np.float32), True

        top_indices = np.argpartition(scores, -n_retrieve)[-n_retrieve:]
        sorted_top_indices = top_indices[np.argsort(-scores[top_indices])]

        candidates = [self.idx2movie[i] for i in sorted_top_indices]
        candidate_scores = scores[sorted_top_indices]

        if len(candidates) < top_n:
            watched_set = self.user_watched_train.get(user_id, set())
            existing = set(candidates)
            filler = [m for m in self.popular_fallback_movies if m not in watched_set and m not in existing]
            needed = top_n - len(candidates)
            candidates.extend(filler[:needed])
            candidate_scores = np.concatenate([candidate_scores, np.zeros(min(needed, len(filler)), dtype=np.float32)])

        return candidates, candidate_scores, False

    def get_candidate_dataframe(
        self,
        user_id: int,
        top_n: int = 300,
        exclude_watched: bool = True
    ) -> pd.DataFrame:
        """Xuất danh sách ứng viên chuẩn schema giao tiếp cho Phase 3."""
        candidates, scores, is_fallback = self.get_top_candidates(
            user_id=user_id,
            top_n=top_n,
            exclude_watched=exclude_watched
        )
        return pd.DataFrame({
            "userId": user_id,
            "movieId": candidates,
            "candidate_score": scores,
            "candidate_source": "content_based_fallback" if is_fallback else "content_based_filtering"
        })

    def save(self, output_dir: Optional[Path] = None) -> None:
        """Lưu trữ Movie Feature Matrix, User Profiles và Mappings vào thư mục chỉ định."""
        if not self.is_fitted:
            raise ValueError("Mô hình chưa được huấn luyện, không thể lưu.")

        save_dir = Path(output_dir or OUTPUT_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Đang lưu các artifacts của Content-Based Filtering vào {save_dir}...")
        np.savez_compressed(
            str(save_dir / "movie_feature_matrix.npz"),
            movie_features=self.movie_feature_matrix,
            feature_names=np.array(self.feature_names)
        )
        np.savez_compressed(
            str(save_dir / "user_profiles.npz"),
            user_profiles=self.user_profiles
        )
        mappings_data = {
            "movie2idx": self.movie2idx,
            "idx2movie": self.idx2movie,
            "user2idx": self.user2idx,
            "idx2user": self.idx2user,
            "popular_fallback_movies": self.popular_fallback_movies
        }
        with open(save_dir / "content_based_mappings.pkl", "wb") as f:
            pickle.dump(mappings_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(save_dir / "content_based_config.json", "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
        with open(save_dir / "content_based_metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.fit_metrics, f, indent=2, ensure_ascii=False)

        logger.info("Đã lưu hoàn tất: movie_feature_matrix.npz, user_profiles.npz, content_based_mappings.pkl, content_based_config.json, content_based_metrics.json")

    def load(self, model_dir: Optional[Path] = None, df_train: Optional[pd.DataFrame] = None) -> "ContentBasedRecommender":
        """Nạp các artifacts từ đĩa."""
        load_dir = Path(model_dir or OUTPUT_DIR)
        mf_path = load_dir / "movie_feature_matrix.npz"
        up_path = load_dir / "user_profiles.npz"
        map_path = load_dir / "content_based_mappings.pkl"
        cfg_path = load_dir / "content_based_config.json"

        if not mf_path.exists() or not up_path.exists() or not map_path.exists():
            raise FileNotFoundError(f"Thiếu artifacts tại {load_dir}. Vui lòng huấn luyện mô hình trước.")

        logger.info(f"Đang nạp Content-Based Filtering artifacts từ {load_dir}...")

        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.config = ContentBasedConfig.from_dict(json.load(f))

        with np.load(str(mf_path)) as data:
            self.movie_feature_matrix = data["movie_features"]
            self.feature_names = list(data["feature_names"])

        with np.load(str(up_path)) as data:
            self.user_profiles = data["user_profiles"]

        with open(map_path, "rb") as f:
            mappings_data = pickle.load(f)
            self.movie2idx = mappings_data["movie2idx"]
            self.idx2movie = mappings_data["idx2movie"]
            self.user2idx = mappings_data["user2idx"]
            self.idx2user = mappings_data["idx2user"]
            self.popular_fallback_movies = mappings_data.get("popular_fallback_movies", [])

        if df_train is not None:
            self.user_watched_train = df_train.groupby("userId")["movieId"].apply(set).to_dict()

        self.is_fitted = True
        logger.info(f"Nạp mô hình thành công! Ma trận phim: {self.movie_feature_matrix.shape}, Hồ sơ user: {self.user_profiles.shape}")
        return self


def main():
    """Chạy độc lập pipeline Content-Based Filtering."""
    import argparse

    parser = argparse.ArgumentParser(description="Triển khai Content-Based Filtering Candidate Retrieval")
    parser.add_argument("--w-genre", type=float, default=0.45, help="Trọng số thể loại")
    parser.add_argument("--w-tag", type=float, default=0.45, help="Trọng số tags TF-IDF")
    parser.add_argument("--w-year", type=float, default=0.10, help="Trọng số năm")
    parser.add_argument("--lambda-neg", type=float, default=0.10, help="Hệ số phạt phản hồi tiêu cực")
    parser.add_argument("--rebuild", action="store_true", help="Buộc tính toán lại")
    args = parser.parse_args()

    train_path = DATA_DIR / "train_interactions.csv.gz"
    movies_path = DATA_DIR / "movies_processed.csv.gz"
    stats_path = DATA_DIR / "movie_stats.csv.gz"
    mf_file = OUTPUT_DIR / "movie_feature_matrix.npz"

    if not train_path.exists() or not movies_path.exists():
        logger.error("Không tìm thấy dữ liệu đã tiền xử lý. Vui lòng chạy phase 1 trước.")
        sys.exit(1)

    logger.info("Đang nạp dữ liệu phim và tương tác huấn luyện...")
    df_train = pd.read_csv(train_path, usecols=["userId", "movieId", "rating", "target"])
    df_movies = pd.read_csv(movies_path)
    df_stats = pd.read_csv(stats_path, usecols=["movieId", "movie_rating_count"]) if stats_path.exists() else None

    config = ContentBasedConfig(
        w_genre=args.w_genre,
        w_tag=args.w_tag,
        w_year=args.w_year,
        neg_penalty_lambda=args.lambda_neg
    )
    recommender = ContentBasedRecommender(config)

    if mf_file.exists() and not args.rebuild:
        logger.info("Phát hiện artifacts sẵn có. Đang nạp...")
        recommender.load(OUTPUT_DIR, df_train=df_train)
    else:
        logger.info("Bắt đầu huấn luyện mô hình mới...")
        recommender.fit(df_train, df_movies, df_movie_stats=df_stats)
        recommender.save(OUTPUT_DIR)
    sample_users = df_train["userId"].unique()[:10]
    sample_dfs = []
    for u in sample_users:
        df_u = recommender.get_candidate_dataframe(int(u), top_n=10)
        sample_dfs.append(df_u)
    sample_export = pd.concat(sample_dfs, ignore_index=True)
    csv_path = OUTPUT_DIR / "content_based_candidates.csv"
    sample_export.to_csv(csv_path, index=False)
    logger.info(f"Đã xuất dữ liệu mẫu 10 người dùng ra {csv_path}")
    logger.info("🎉 HOÀN TẤT HUẤN LUYỆN VÀ LƯU TRỮ CONTENT-BASED FILTERING!")


if __name__ == "__main__":
    main()

