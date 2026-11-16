import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


DATA_DIR = Path(__file__).resolve().parent / "dataset"
MAX_CANDIDATES = 300
GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
          "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def find_file(directory, *names, required=True):
    """Tìm CSV thường hoặc CSV nén trong thư mục dữ liệu."""
    for name in names:
        path = Path(directory) / name
        if path.is_file():
            return path
    if required:
        raise FileNotFoundError(f"Thiếu dữ liệu: {' hoặc '.join(names)}")
    return None


class MovieRecommender:
    """Nạp mô hình một lần và dùng chung cho giao diện, dòng lệnh, đánh giá."""

    def __init__(self, data_dir=DATA_DIR, progress=print):
        self.data_dir = Path(data_dir)
        self.progress = progress
        progress("Đang nạp mô hình CF và danh mục phim...")
        self.user_vectors = np.load(find_file(data_dir, "user_latent.npy"), allow_pickle=False)
        self.movie_vectors = np.load(find_file(data_dir, "movie_latent.npy"), allow_pickle=False)
        if self.user_vectors.ndim != 2 or self.movie_vectors.ndim != 2 or self.user_vectors.shape[1] != self.movie_vectors.shape[1]:
            raise ValueError("Hai ma trận CF không cùng số chiều; hãy chạy lại giai đoạn 2")
        if not len(self.user_vectors) or not len(self.movie_vectors) or not self.user_vectors.shape[1]:
            raise ValueError("Ma trận CF không được rỗng")
        # Chỉ nạp bộ ánh xạ do giai đoạn 2 tạo trong thư mục dự án.
        with find_file(data_dir, "cf_mappings.pkl").open("rb") as stream:
            mapping = pickle.load(stream)
        self.user_ids = self._ordered_ids(mapping["user2idx"], len(self.user_vectors))
        self.movie_ids = self._ordered_ids(mapping["movie2idx"], len(self.movie_vectors))
        if not np.isfinite(self.user_vectors).all() or not np.isfinite(self.movie_vectors).all():
            raise ValueError("Ma trận CF có giá trị không hợp lệ")
        self.users = pd.Index(self.user_ids)
        self.items = pd.Index(self.movie_ids)
        movie_path = find_file(data_dir, "movies_processed.csv", "movies_processed.csv.gz",
                               "ml-32m/movies.csv", "movies.csv")
        self.movies = pd.read_csv(movie_path).set_index("movieId")
        if not self.movies.index.is_unique:
            raise ValueError("Danh mục phim chứa movieId trùng lặp")
        if "clean_title" not in self.movies:
            self.movies["clean_title"] = self.movies["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True).str.strip()
        if "release_year" not in self.movies:
            years = pd.to_numeric(self.movies["title"].str.extract(r"\((\d{4})\)", expand=False), errors="coerce")
            self.movies["release_year"] = years.fillna(years.median() if years.notna().any() else 0).astype(int)
        self.movies["genres"] = self.movies["genres"].fillna("(no genres listed)")
        if not self.items.isin(self.movies.index).all():
            raise ValueError("Danh mục phim không khớp bộ ánh xạ CF")
        self.genres = sorted(set(self.movies["genres"].str.split("|").explode()) - {"(no genres listed)"})
        self.train_path = find_file(data_dir, "train_interactions.csv", "train_interactions.csv.gz",
                                    "phase2/train_interactions.csv", "phase2/train_interactions.csv.gz")
        self.test_path = find_file(self.train_path.parent, "test_interactions.csv", "test_interactions.csv.gz", required=False)
        self._load_history()
        self.ranker = None
        self.feature_columns = []
        self._load_ranker()
        metrics_path = self.data_dir / "phase3_metrics.json"
        self.notebook_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        progress("Đã sẵn sàng gợi ý phim.")

    @staticmethod
    def _ordered_ids(mapping, size):
        """Đảm bảo thứ tự ID khớp chính xác với hàng vector đã lưu."""
        if len(mapping) != size or sorted(mapping.values()) != list(range(size)):
            raise ValueError("Bộ ánh xạ không khớp ma trận CF; hãy chạy lại giai đoạn 2")
        result = np.empty(size, dtype=np.int64)
        for item_id, position in mapping.items():
            result[position] = int(item_id)
        return result

    def _load_history(self):
        """Dùng train để loại phim đã xem; nạp đúng thống kê notebook đã dùng khi học."""
        self.progress("Đang đọc lịch sử train và các bảng thống kê của notebook...")
        rows, columns, ratings = [], [], []
        for chunk in pd.read_csv(self.train_path, usecols=["userId", "movieId", "rating"], chunksize=500_000):
            u = self.users.get_indexer(chunk["userId"])
            m = self.items.get_indexer(chunk["movieId"])
            values = chunk["rating"].to_numpy(dtype=np.float32)
            if (u < 0).any() or (m < 0).any() or not np.isfinite(values).all():
                raise ValueError("Tập huấn luyện không khớp mô hình CF; hãy chạy lại giai đoạn 2")
            rows.append(u.astype(np.int32))
            columns.append(m.astype(np.int32))
            ratings.append(values)
        if not rows or sum(len(values) for values in ratings) == 0:
            raise ValueError("Tập huấn luyện không được rỗng")
        self.history = csr_matrix((np.concatenate(ratings), (np.concatenate(rows), np.concatenate(columns))),
                                  shape=(len(self.users), len(self.items)))
        movie_path = find_file(self.data_dir, "movie_stats.csv", "movie_stats.csv.gz")
        user_path = find_file(self.data_dir, "user_stats.csv", "user_stats.csv.gz")
        movie_stats = pd.read_csv(movie_path).set_index("movieId")
        self.user_stats = pd.read_csv(user_path).set_index("userId")
        self.global_rating = float(movie_stats["movie_avg_rating"].mean())
        self.movies = self.movies.join(movie_stats[["movie_avg_rating", "movie_rating_count", "movie_log_count"]])
        self.movies["movie_rating_count"] = self.movies["movie_rating_count"].fillna(0).astype(int)

    def _load_ranker(self):
        model_path = self.data_dir / "lgbm_ranker.txt"
        columns_path = self.data_dir / "feature_cols.pkl"
        if model_path.exists() != columns_path.exists():
            raise ValueError("Cần đủ lgbm_ranker.txt và feature_cols.pkl; hãy huấn luyện lại mô hình xếp hạng")
        if not model_path.exists():
            return
        import lightgbm as lgb

        with columns_path.open("rb") as stream:
            self.feature_columns = pickle.load(stream)
        self.ranker = lgb.Booster(model_file=str(model_path))
        if self.ranker.feature_name() != self.feature_columns:
            raise ValueError("Danh sách đặc trưng không khớp mô hình LightGBM")
        if self.ranker.params.get("objective") != "binary":
            raise ValueError("Demo xác suất thích cần mô hình LightGBM phân loại nhị phân theo notebook giai đoạn 3")
        available = set(self.movies.columns) | {"cf_score", "user_avg_rating", "user_log_count", "rating_diff_from_user_avg"}
        missing = set(self.feature_columns) - available
        if missing:
            raise ValueError(f"Thiếu đặc trưng: {', '.join(sorted(missing))}. Hãy chạy ô xuất kết quả ở cuối notebook giai đoạn 3.")

    def build_features(self, user_ids, movie_ids):
        """Ghép đúng các đặc trưng đã dùng lúc huấn luyện, giữ nguyên thứ tự dòng."""
        user_ids, movie_ids = np.asarray(user_ids), np.asarray(movie_ids)
        if len(user_ids) != len(movie_ids) or not pd.Index(movie_ids).isin(self.movies.index).all():
            raise ValueError("Cặp người dùng–phim không hợp lệ")
        columns = ["release_year", "movie_age"] + [f"genre_{genre}" for genre in GENRES] + ["genre_count"]
        columns += sorted(column for column in self.movies if column.startswith("tag_"))
        columns += ["movie_avg_rating", "movie_log_count"]
        result = self.movies.reindex(movie_ids)[columns].reset_index(drop=True)
        stats = self.user_stats.reindex(user_ids).reset_index(drop=True)
        result["user_avg_rating"] = stats["user_avg_rating"]
        result["user_log_count"] = stats["user_log_count"]
        # Giữ cùng thứ tự tính và điền giá trị thiếu như build_features trong notebook.
        result["rating_diff_from_user_avg"] = (result["movie_avg_rating"] - result["user_avg_rating"]).fillna(0)
        result["movie_avg_rating"] = result["movie_avg_rating"].fillna(self.global_rating)
        result["movie_log_count"] = result["movie_log_count"].fillna(0)
        u, m = self.users.get_indexer(user_ids), self.items.get_indexer(movie_ids)
        scores = np.zeros(len(result), dtype=np.float32)
        valid = np.flatnonzero((u >= 0) & (m >= 0))
        for start in range(0, len(valid), 50_000):
            indices = valid[start:start + 50_000]
            scores[indices] = np.sum(self.user_vectors[u[indices]] * self.movie_vectors[m[indices]], axis=1)
        result.insert(0, "cf_score", scores)
        return result

    def get_candidates(self, user_id, genre=None):
        """Vòng 1: lấy tối đa 300 ứng viên CF, loại mọi phim đã có trong train."""
        if isinstance(user_id, bool) or not isinstance(user_id, (int, np.integer)) or user_id < 1:
            raise ValueError("ID người dùng phải là số nguyên dương")
        if genre is not None and genre not in self.genres:
            raise ValueError("Thể loại phim không hợp lệ")
        position = self.users.get_indexer([user_id])[0]
        if position >= 0:
            scores = self.user_vectors[position] @ self.movie_vectors.T
            begin, end = self.history.indptr[position:position + 2]
            scores[self.history.indices[begin:end]] = -np.inf
            candidates = pd.DataFrame({"movieId": self.movie_ids, "cf_score": scores})
            candidates = candidates[np.isfinite(candidates["cf_score"])]
        else:
            # Người dùng mới bắt đầu từ phim phổ biến; chưa có vector sở thích cá nhân.
            candidates = self.movies.reset_index()[["movieId", "movie_log_count"]].rename(columns={"movie_log_count": "retrieval_score"})
            candidates["cf_score"] = 0.0
        if "retrieval_score" not in candidates:
            candidates["retrieval_score"] = candidates["cf_score"]
        if genre is not None:
            matching = self.movies.index[self.movies["genres"].str.split("|").apply(lambda values: genre in values)]
            candidates = candidates[candidates["movieId"].isin(matching)]
        candidates = (candidates.sort_values(["retrieval_score", "movieId"], ascending=[False, True])
                      .head(MAX_CANDIDATES).drop(columns="retrieval_score").reset_index(drop=True))
        candidates.insert(0, "cf_rank", np.arange(1, len(candidates) + 1))
        return candidates.join(self.movies[["clean_title", "release_year", "genres", "movie_avg_rating"]], on="movieId")

    def rank_candidates(self, user_id, candidates, top_k=10, hybrid=True):
        """Vòng 2: dùng đúng đặc trưng/mô hình giai đoạn 3 để xếp hạng ứng viên."""
        if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or not 1 <= top_k <= 50:
            raise ValueError("Số phim gợi ý phải từ 1 đến 50")
        if hybrid and self.ranker is None:
            raise ValueError("Chưa có LightGBM; hãy chạy notebook giai đoạn 3 hoặc chọn chế độ chỉ CF")
        # Sao chép để giữ nguyên danh sách vòng 1 cho bảng so sánh trên giao diện.
        ranked = candidates.copy()
        ranked["predicted_like_prob"] = np.nan
        if hybrid and not ranked.empty:
            features = self.build_features(np.full(len(ranked), user_id), ranked["movieId"].to_numpy())
            missing = set(self.feature_columns) - set(features.columns)
            if missing:
                raise ValueError(f"Thiếu đặc trưng cho LightGBM: {', '.join(sorted(missing))}")
            ranked["predicted_like_prob"] = self.ranker.predict(features[self.feature_columns], num_threads=4)
            ranked["cf_score"] = features["cf_score"].to_numpy()
            ranked = ranked.sort_values(["predicted_like_prob", "cf_score", "movieId"], ascending=[False, False, True])
        result = ranked.head(top_k).drop(columns="cf_rank").copy()
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        result.insert(0, "userId", user_id)
        columns = ["userId", "rank", "movieId", "cf_score", "predicted_like_prob",
                   "clean_title", "release_year", "genres", "movie_avg_rating"]
        return result[columns].reset_index(drop=True)

    def recommend(self, user_id, top_k=10, genre=None, hybrid=True):
        """Luồng hoàn chỉnh: người dùng → 300 ứng viên → xếp hạng → Top K phim."""
        candidates = self.get_candidates(user_id, genre)
        return self.rank_candidates(user_id, candidates, top_k, hybrid)

    def get_history(self, user_id, limit=100):
        """Xem các phim đã đánh giá, ưu tiên điểm cao; dữ liệu chỉ lấy từ train."""
        position = self.users.get_indexer([user_id])[0]
        if position < 0:
            return pd.DataFrame(columns=["movieId", "rating", "clean_title", "release_year", "genres"])
        row = self.history.getrow(position)
        result = pd.DataFrame({"movieId": self.movie_ids[row.indices], "rating": row.data})
        return (result.sort_values(["rating", "movieId"], ascending=[False, True]).head(limit)
                .join(self.movies[["clean_title", "release_year", "genres"]], on="movieId").reset_index(drop=True))

    def evaluate(self, max_users=100, top_k=10, hybrid=True):
        """Đánh giá danh sách cuối cùng trên phim thích chưa xem trong tập test."""
        if self.test_path is None:
            raise FileNotFoundError("Chưa có tập kiểm tra để đánh giá")
        if max_users < 1:
            raise ValueError("Số người dùng đánh giá phải lớn hơn 0")
        test = pd.read_csv(self.test_path, usecols=["userId", "movieId", "target"])
        test = test[(test["target"] == 1) & test["userId"].isin(self.users) & test["movieId"].isin(self.movies.index)].drop_duplicates()
        known_movies = self.items.get_indexer(test["movieId"])
        known_users = self.users.get_indexer(test["userId"])
        seen = np.zeros(len(test), dtype=bool)
        known = known_movies >= 0
        seen[known] = np.asarray(self.history[known_users[known], known_movies[known]]).ravel() != 0
        truth = test.loc[~seen].groupby("userId")["movieId"].agg(set).head(max_users)
        recalls, precisions, hits, ndcgs = [], [], [], []
        for index, (user_id, liked) in enumerate(truth.items(), start=1):
            recommendations = self.recommend(int(user_id), top_k=top_k, hybrid=hybrid)
            relevance = recommendations["movieId"].isin(liked).to_numpy(dtype=float)
            count = int(relevance.sum())
            recalls.append(count / len(liked))
            precisions.append(count / len(recommendations) if len(recommendations) else 0.0)
            hits.append(count > 0)
            ideal = np.sum(1 / np.log2(np.arange(min(top_k, len(liked))) + 2))
            ndcgs.append(float(np.sum(relevance / np.log2(np.arange(len(relevance)) + 2)) / ideal))
            if index % 20 == 0:
                self.progress(f"Đã đánh giá {index}/{len(truth)} người dùng...")
        return {"users": len(truth), "k": top_k, "mode": "hybrid" if hybrid else "cf",
                "statistics_scope": self.notebook_metrics.get("statistics_scope", "notebook_export"),
                "recall": float(np.mean(recalls)) if recalls else 0.0,
                "precision": float(np.mean(precisions)) if precisions else 0.0,
                "hit_rate": float(np.mean(hits)) if hits else 0.0,
                "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Thư mục dữ liệu và mô hình")
    parser.add_argument("--user-id", type=int, help="ID người dùng muốn gợi ý")
    parser.add_argument("--top-k", type=int, default=10, help="Số phim gợi ý, từ 1 đến 50")
    parser.add_argument("--evaluate-users", type=int, default=0, help="Số người dùng test để đánh giá")
    args = parser.parse_args()
    engine = MovieRecommender(args.data_dir)
    user_id = args.user_id if args.user_id is not None else int(engine.user_ids[0])
    print(f"\nGợi ý cho người dùng {user_id} bằng {'CF + LightGBM' if engine.ranker else 'CF'}:")
    result = engine.recommend(user_id, args.top_k, hybrid=engine.ranker is not None)
    print(result.rename(columns={
        "userId": "Người dùng", "rank": "Hạng", "movieId": "ID phim", "cf_score": "Điểm CF",
        "predicted_like_prob": "Xác suất thích", "clean_title": "Tên phim", "release_year": "Năm",
        "genres": "Thể loại", "movie_avg_rating": "Điểm TB train",
    }).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    if args.evaluate_users:
        metrics = engine.evaluate(args.evaluate_users, args.top_k, hybrid=engine.ranker is not None)
        print(f"Số người dùng được đánh giá: {metrics['users']}")
        for label, key in [("Độ thu hồi", "recall"), ("Độ chính xác", "precision"),
                           ("Tỷ lệ gợi ý trúng", "hit_rate"), ("Chất lượng thứ hạng (NDCG)", "ndcg")]:
            print(f"{label}@{metrics['k']}: {metrics[key]:.2%}")
        (args.data_dir / "phase4_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import sys

    for output in (sys.stdout, sys.stderr):
        if hasattr(output, "reconfigure"):
            output.reconfigure(encoding="utf-8")
    main()
