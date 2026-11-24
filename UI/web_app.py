"""Website gợi ý phim, dùng lại mô hình và logic suy luận trong final.py."""

import argparse
import hashlib
import json
import os
import pickle
import random
import socket
import threading
import time
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

import numpy as np
import pandas as pd
import lightgbm as lgb
from flask import Flask, jsonify, redirect, render_template, request
from werkzeug.exceptions import HTTPException, NotFound, ServiceUnavailable
from werkzeug.serving import ThreadedWSGIServer

from code_danh_gia import user_metrics
from final import HybridMovieRecommender
from phase4_recommender import DATA_DIR, MovieRecommender
from UI.posters import PosterCatalog


class BoosterClassifier:
    """Giao diện predict_proba cho Booster LightGBM đã lưu trên đĩa."""

    def __init__(self, booster):
        self.booster = booster

    def predict_proba(self, features):
        probabilities = self.booster.predict(features, num_threads=4)
        return np.column_stack((1 - probabilities, probabilities))


class WatchedHistory:
    """Lấy phim đã xem từ ma trận thưa, không nhân bản toàn bộ lịch sử."""

    def __init__(self, engine):
        self.engine = engine

    def get(self, user_id, default=None):
        position = self.engine.users.get_indexer([user_id])[0]
        if position < 0:
            return default
        begin, end = self.engine.history.indptr[position:position + 2]
        return set(self.engine.movie_ids[self.engine.history.indices[begin:end]])


class SavedArtifacts:
    """Đọc trọn bộ đầu vào từ final.py, không trộn thống kê của notebook cũ."""

    def __init__(self, data_dir):
        data_dir = Path(data_dir)
        model_path = data_dir / 'lgbm_ranker.txt'
        columns_path = data_dir / 'feature_cols.pkl'
        if not model_path.is_file() or not columns_path.is_file():
            raise FileNotFoundError('Thiếu mô hình. Hãy chạy final.py trước khi mở website.')
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        bundle_path = data_dir / 'serving_artifacts.pkl'
        bundle = None
        if bundle_path.is_file():
            with bundle_path.open('rb') as stream:
                bundle = pickle.load(stream)
        if bundle is None:
            raise FileNotFoundError(
                'Thiếu serving_artifacts.pkl. Hãy chạy final.py với cùng --data-dir '
                'để lưu đồng bộ mô hình và dữ liệu suy luận.'
            )
        if bundle.get('version') != 1 or bundle.get('model_sha256') != model_hash:
            raise ValueError('Bộ dữ liệu suy luận không khớp mô hình; hãy chạy lại final.py với cùng --data-dir.')
        self.user_vectors = bundle['user_latent']
        self.movie_vectors = bundle['movie_latent']
        if (self.user_vectors.ndim != 2 or self.movie_vectors.ndim != 2
                or not self.user_vectors.size or not self.movie_vectors.size
                or self.user_vectors.shape[1] != self.movie_vectors.shape[1]
                or not np.isfinite(self.user_vectors).all() or not np.isfinite(self.movie_vectors).all()):
            raise ValueError('Ma trận CF không hợp lệ; hãy chạy lại final.py.')
        self.user_ids = MovieRecommender._ordered_ids(bundle['user2idx'], len(self.user_vectors))
        self.movie_ids = MovieRecommender._ordered_ids(bundle['movie2idx'], len(self.movie_vectors))
        self.users = pd.Index(self.user_ids)
        self.items = pd.Index(self.movie_ids)
        self.history = bundle['history']
        if self.history.shape != (len(self.users), len(self.items)):
            raise ValueError('Lịch sử không khớp mô hình; hãy chạy lại final.py.')
        self.movies = bundle['df_movies_final'].set_index('movieId')
        self.movies = self.movies.join(bundle['df_movie_stats'].set_index('movieId'))
        self.movies['movie_rating_count'] = self.movies['movie_rating_count'].fillna(0).astype(int)
        self.user_stats = bundle['df_user_stats'].set_index('userId')
        self.genres = sorted(set(self.movies['genres'].fillna('').str.split('|').explode()) - {'', '(no genres listed)'})
        self.feature_columns = bundle['feature_cols']
        self.ranker = lgb.Booster(model_file=str(model_path))
        with columns_path.open('rb') as stream:
            saved_columns = pickle.load(stream)
        if self.ranker.feature_name() != self.feature_columns or saved_columns != self.feature_columns:
            raise ValueError('Đặc trưng không khớp mô hình; hãy chạy lại final.py.')
        print('Đã nạp bộ dữ liệu và mô hình của final.py.', flush=True)

    def get_history(self, user_id, limit=8, offset=0):
        position = self.users.get_indexer([user_id])[0]
        if position < 0:
            raise NotFound('Không tìm thấy người dùng trong mô hình.')
        row = self.history.getrow(position)
        result = pd.DataFrame({'movieId': self.movie_ids[row.indices], 'rating': row.data})
        return (result.sort_values(['rating', 'movieId'], ascending=[False, True]).iloc[offset:offset + limit]
                .join(self.movies[['clean_title', 'release_year', 'genres']], on='movieId').reset_index(drop=True))


class RecommendationService:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = Path(data_dir)
        self.engine = SavedArtifacts(data_dir)
        statistics = ['movie_avg_rating', 'movie_rating_count', 'movie_log_count']
        self.recommender = HybridMovieRecommender(
            user_latent=self.engine.user_vectors,
            movie_latent=self.engine.movie_vectors,
            user2idx={int(value): index for index, value in enumerate(self.engine.user_ids)},
            movie2idx={int(value): index for index, value in enumerate(self.engine.movie_ids)},
            idx2movie={index: int(value) for index, value in enumerate(self.engine.movie_ids)},
            df_movies_final=self.engine.movies.drop(columns=statistics).reset_index(),
            df_movie_stats=self.engine.movies[statistics].reset_index(),
            df_user_stats=self.engine.user_stats.reset_index(),
            lgb_model=BoosterClassifier(self.engine.ranker),
            feature_cols=self.engine.feature_columns,
            user_watched_train=WatchedHistory(self.engine),
        )
        self.catalog = self.engine.movies.loc[self.engine.items, [
            'clean_title', 'release_year', 'genres', 'movie_avg_rating', 'movie_rating_count',
        ]].sort_values(['movie_rating_count', 'clean_title'], ascending=[False, True])
        self.catalog['clean_title'] = self.catalog['clean_title'].str.replace(r'^(.+), (The|An|A)$', r'\2 \1', regex=True)
        self.lock = threading.Lock()
        self.evaluation_lock = threading.Lock()
        self._evaluation_truth = None
        self._evaluation_signature = None

    @staticmethod
    def records(frame):
        # pandas chuyển cả số numpy và giá trị thiếu về JSON hợp lệ.
        if 'clean_title' in frame:
            frame = frame.copy()
            frame['clean_title'] = frame['clean_title'].str.replace(r'^(.+), (The|An|A)$', r'\2 \1', regex=True)
        return json.loads(frame.to_json(orient='records', force_ascii=False))

    def movies(self, query='', genre='', limit=12, ids=None):
        return self.catalog_page(query, genre, limit, ids)['movies']

    def catalog_page(self, query='', genre='', limit=12, ids=None, offset=0):
        movies = self.catalog
        if ids is not None:
            movies = movies.reindex(list(dict.fromkeys(value for value in ids if value in movies.index)))
        if query:
            movies = movies[movies['clean_title'].str.contains(query, case=False, regex=False, na=False)]
        if genre:
            if genre not in self.engine.genres:
                raise ValueError('Thể loại phim không hợp lệ.')
            movies = movies[movies['genres'].str.split('|').apply(lambda values: genre in values)]
        return {
            'movies': self.records(movies.iloc[offset:offset + limit].reset_index()),
            'total': len(movies), 'limit': limit, 'offset': offset,
            'has_more': offset + limit < len(movies),
        }

    def movie(self, movie_id):
        if movie_id not in self.catalog.index:
            raise NotFound('Không tìm thấy phim trong kho gợi ý.')
        return self.records(self.catalog.loc[[movie_id]].reset_index())[0]

    def user_history(self, user_id, limit=20, offset=0):
        position = self.engine.users.get_indexer([user_id])[0]
        if position < 0:
            raise NotFound('Không tìm thấy người dùng trong mô hình.')
        begin, end = self.engine.history.indptr[position:position + 2]
        total = int(end - begin)
        return {
            'user_id': user_id,
            'movies': self.records(self.engine.get_history(user_id, limit, offset)),
            'total': total, 'limit': limit, 'offset': offset,
            'has_more': offset + limit < total,
        }

    def recommend(self, payload):
        top_k = integer(payload.get('top_k', 10), 'Số phim', 1, 50)
        mode = payload.get('mode', 'new')
        candidate_count = min(300, len(self.engine.movie_ids))
        history = []
        started = time.perf_counter()
        with self.lock:
            if mode == 'new':
                picked = payload.get('liked_movie_ids')
                if not isinstance(picked, list) or not 1 <= len(picked) <= 20:
                    raise ValueError('Hãy chọn từ 1 đến 20 phim bạn yêu thích.')
                picked = list(dict.fromkeys(integer(value, 'ID phim', 1) for value in picked))
                if any(value not in self.recommender.movie2idx for value in picked):
                    raise ValueError('Phim đã chọn chưa có trong mô hình. Hãy chọn phim từ ô tìm kiếm.')
                result = self.recommender.recommend_for_new_user(picked, top_k, candidate_count)
                context = {'mode': mode, 'liked_count': len(picked)}
            elif mode == 'existing':
                user_id = integer(payload.get('user_id'), 'ID người dùng', 1)
                if user_id not in self.recommender.user2idx:
                    raise ValueError('ID này chưa có lịch sử. Hãy chọn ID ngẫu nhiên hoặc dùng chế độ chọn phim.')
                result = self.recommender.recommend_for_user(user_id, top_k, candidate_count)
                history = self.records(self.engine.get_history(user_id, limit=8))
                context = {'mode': mode, 'user_id': user_id}
            else:
                raise ValueError('Chế độ gợi ý không hợp lệ.')
        return {
            'movies': self.records(result), 'context': context, 'history': history,
            'elapsed_ms': round((time.perf_counter() - started) * 1000),
        }

    def compare(self, payload):
        """So sánh cùng sở thích và cùng kho phim; điểm CF không phải xác suất."""
        hybrid = self.recommend(payload)
        top_k = integer(payload.get('top_k', 10), 'Số phim', 1, 50)
        if hybrid['context']['mode'] == 'existing':
            user_id = hybrid['context']['user_id']
            user_vector = self.engine.user_vectors[self.recommender.user2idx[user_id]]
            excluded = self.recommender.user_watched.get(user_id, set())
        else:
            picked = list(dict.fromkeys(payload['liked_movie_ids']))
            excluded = set(picked)
            positions = [self.recommender.movie2idx[movie_id] for movie_id in picked]
            user_vector = self.engine.movie_vectors[positions].mean(axis=0)

        started = time.perf_counter()
        candidate_ids = self.recommender._candidate_ids(user_vector, excluded, min(300, len(self.catalog)))[:top_k]
        cf_movies = self.catalog.reindex(candidate_ids).copy()
        positions = [self.recommender.movie2idx[movie_id] for movie_id in candidate_ids]
        cf_movies['cf_score'] = user_vector.dot(self.engine.movie_vectors[positions].T)
        cf_elapsed = round((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        popular = self.catalog.loc[~self.catalog.index.isin(excluded)].head(top_k)
        popularity_elapsed = round((time.perf_counter() - started) * 1000)
        return {
            'methods': [
                {'id': 'hybrid', 'name': 'Hybrid CF + LightGBM', 'movies': hybrid['movies'],
                 'elapsed_ms': hybrid['elapsed_ms']},
                {'id': 'cf', 'name': 'Collaborative Filtering', 'movies': self.records(cf_movies.reset_index()),
                 'elapsed_ms': cf_elapsed},
                {'id': 'popularity', 'name': 'Phổ biến (Popularity)',
                 'movies': self.records(popular.reset_index()), 'elapsed_ms': popularity_elapsed},
            ],
            'context': hybrid['context'], 'top_k': top_k, 'history': hybrid['history'],
        }

    def _test_path(self):
        for name in ('test_interactions.csv', 'test_interactions.csv.gz'):
            candidate = self.data_dir / name
            if candidate.is_file():
                return candidate
        return None

    def evaluation_config(self):
        available = self._test_path() is not None
        return {
            'available': available, 'default_users': 20, 'max_users': 100,
            'reason': '' if available else 'Chưa có test_interactions.csv hoặc test_interactions.csv.gz trong thư mục dữ liệu.',
        }

    def _test_positives(self, test_path):
        """Đọc test theo khối; chỉ giữ phim thích chưa xuất hiện trong train."""
        stat = test_path.stat()
        signature = (str(test_path.resolve()), stat.st_size, stat.st_mtime_ns)
        if self._evaluation_signature == signature:
            return self._evaluation_truth
        columns = set(pd.read_csv(test_path, nrows=0).columns)
        if not {'userId', 'movieId'}.issubset(columns) or not {'target', 'rating'}.intersection(columns):
            raise ValueError('Tập test cần userId, movieId và target hoặc rating.')
        label = 'target' if 'target' in columns else 'rating'
        truth = {}
        for chunk in pd.read_csv(test_path, usecols=['userId', 'movieId', label], chunksize=250_000):
            labels = pd.to_numeric(chunk[label], errors='coerce')
            chunk = chunk.loc[labels.eq(1) if label == 'target' else labels.ge(4)]
            users = self.engine.users.get_indexer(chunk['userId'])
            movies = self.engine.items.get_indexer(chunk['movieId'])
            valid = (users >= 0) & (movies >= 0)
            if not valid.any():
                continue
            chunk = chunk.loc[valid]
            watched = np.asarray(self.engine.history[users[valid], movies[valid]]).ravel() != 0
            for user_id, group in chunk.loc[~watched].groupby('userId'):
                truth.setdefault(int(user_id), set()).update(int(movie_id) for movie_id in group['movieId'])
        self._evaluation_truth = truth
        self._evaluation_signature = signature
        return truth

    def evaluate(self, payload):
        """Đánh giá cả ba cách trên đúng cùng nhóm người dùng và ground truth."""
        top_k = integer(payload.get('top_k', 10), 'Số phim', 1, 50)
        max_users = integer(payload.get('max_users', 20), 'Số người dùng đánh giá', 1, 100)
        test_path = self._test_path()
        if test_path is None:
            raise ServiceUnavailable(self.evaluation_config()['reason'])
        started = time.perf_counter()
        with self.evaluation_lock:
            truth = self._test_positives(test_path)
            if not truth:
                raise ValueError('Tập test không có người dùng với phim yêu thích chưa xem và có trong mô hình.')
            users = np.random.default_rng(42).choice(sorted(truth), min(max_users, len(truth)), replace=False)
            results = {}
            recommended_ids = {}
            for user_id in users:
                comparison = self.compare({'mode': 'existing', 'user_id': int(user_id), 'top_k': top_k})
                for method in comparison['methods']:
                    method_id = method['id']
                    ids = [movie['movieId'] for movie in method['movies']]
                    metrics = user_metrics(ids, truth[int(user_id)], ks=(top_k,))
                    if method_id not in results:
                        results[method_id] = {'id': method_id, 'name': method['name'],
                                              'precision': [], 'recall': [], 'ndcg': [], 'hit_rate': []}
                        recommended_ids[method_id] = set()
                    for field, metric in (('precision', 'P'), ('recall', 'R'), ('ndcg', 'NDCG'), ('hit_rate', 'HitRate')):
                        results[method_id][field].append(metrics[f'{metric}@{top_k}'])
                    recommended_ids[method_id].update(ids)
            methods = []
            for method_id, result in results.items():
                methods.append({
                    'id': method_id, 'name': result['name'],
                    **{field: float(np.mean(result[field])) for field in ('precision', 'recall', 'ndcg', 'hit_rate')},
                    'coverage': len(recommended_ids[method_id]) / len(self.catalog),
                })
        return {
            'top_k': top_k, 'user_count': len(users), 'eligible_user_count': len(truth),
            'methods': methods, 'user_ids': [int(user_id) for user_id in users], 'seed': 42,
            'source': test_path.name, 'elapsed_ms': round((time.perf_counter() - started) * 1000),
            'protocol': 'Cả ba phương pháp dùng cùng mẫu người dùng test với seed 42. '
                        'Phim liên quan là target = 1 (hoặc rating ≥ 4), có trong kho và chưa có trong lịch sử train. '
                        'Thống kê, độ phổ biến và sở thích chỉ lấy từ bộ train đã lưu. '
                        'Precision@K chia cho K; các chỉ số được lấy trung bình theo người dùng. '
                        'Coverage là tỷ lệ phim khác nhau được gợi ý trên toàn kho.',
        }


def integer(value, label, minimum, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{label} phải là số nguyên.')
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f' đến {maximum}' if maximum is not None else ' trở lên'
        raise ValueError(f'{label} phải từ {minimum}{suffix}.')
    return value


def create_app(data_dir=DATA_DIR, service=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    service = service if service is not None else RecommendationService(data_dir)
    app.extensions['recommendations'] = service
    app.extensions['posters'] = PosterCatalog(getattr(service, 'data_dir', data_dir))

    @app.url_defaults
    def version_static_assets(endpoint, values):
        if endpoint == 'static' and 'filename' in values and 'v' not in values:
            path = Path(app.static_folder) / values['filename']
            if path.is_file():
                values['v'] = hashlib.sha256(path.read_bytes()).hexdigest()[:12]

    @app.after_request
    def refresh_interface(response):
        if request.endpoint == 'index':
            response.cache_control.no_store = True
        if request.endpoint == 'health':
            response.headers['X-CineMatch-Instance'] = instance_id(getattr(service, 'data_dir', data_dir))
        return response

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/api/config')
    def configuration():
        starters = [value for value in (79132, 2571, 2959) if value in service.catalog.index]
        return jsonify({
            'movie_count': len(service.catalog), 'user_count': len(service.engine.users),
            'genres': service.engine.genres, 'starter_movies': service.movies(ids=starters),
            'default_user_id': int(service.engine.user_ids[0]),
        })

    @app.get('/api/health')
    def health():
        return jsonify({
            'status': 'ready', 'movie_count': len(service.catalog),
            'user_count': len(service.engine.users),
        })

    @app.get('/api/movies')
    def movies():
        query = request.args.get('q', '').strip()
        if len(query) > 150:
            raise ValueError('Từ khóa tìm kiếm tối đa 150 ký tự.')
        try:
            limit = int(request.args.get('limit', '12'))
            offset = int(request.args.get('offset', '0'))
            raw_ids = request.args.get('ids')
            ids = None if raw_ids is None else [int(value) for value in raw_ids.split(',') if value]
        except ValueError:
            raise ValueError('Số phim hoặc ID phim không hợp lệ.') from None
        integer(limit, 'Số phim', 1, 50)
        integer(offset, 'Vị trí bắt đầu', 0)
        if ids is not None and len(ids) > 50:
            raise ValueError('Chỉ có thể nạp tối đa 50 phim mỗi lần.')
        if ids is not None:
            for movie_id in ids:
                integer(movie_id, 'ID phim', 1)
        return jsonify(service.catalog_page(query, request.args.get('genre', ''), limit, ids, offset))

    @app.get('/api/movies/<int:movie_id>')
    def movie_detail(movie_id):
        return jsonify({'movie': service.movie(movie_id)})

    @app.get('/api/movies/<int:movie_id>/poster')
    def movie_poster(movie_id):
        if movie_id not in service.catalog.index:
            raise NotFound('Không tìm thấy phim trong kho gợi ý.')
        poster_url = app.extensions['posters'].poster_url(movie_id)
        if poster_url is None:
            raise NotFound('Chưa có ảnh cho phim này.')
        response = redirect(poster_url, code=302)
        response.cache_control.public = True
        response.cache_control.max_age = 86400
        return response

    @app.get('/api/users/<int:user_id>/history')
    def user_history(user_id):
        try:
            limit = int(request.args.get('limit', '20'))
            offset = int(request.args.get('offset', '0'))
        except ValueError:
            raise ValueError('Số phim hoặc vị trí bắt đầu không hợp lệ.') from None
        integer(limit, 'Số phim', 1, 50)
        integer(offset, 'Vị trí bắt đầu', 0)
        return jsonify(service.user_history(user_id, limit, offset))

    @app.get('/api/users/random')
    def random_user():
        return jsonify({'user_id': int(random.choice(service.engine.user_ids))})

    @app.post('/api/recommend')
    def recommend():
        payload = request.get_json()
        if not isinstance(payload, dict):
            raise ValueError('Dữ liệu gửi lên phải là một đối tượng JSON.')
        return jsonify(service.recommend(payload))

    @app.post('/api/compare')
    def compare():
        payload = request.get_json()
        if not isinstance(payload, dict):
            raise ValueError('Dữ liệu gửi lên phải là một đối tượng JSON.')
        return jsonify(service.compare(payload))

    @app.get('/api/evaluation/config')
    def evaluation_configuration():
        return jsonify(service.evaluation_config())

    @app.post('/api/evaluate')
    def evaluate():
        payload = request.get_json()
        if not isinstance(payload, dict):
            raise ValueError('Dữ liệu gửi lên phải là một đối tượng JSON.')
        return jsonify(service.evaluate(payload))

    @app.errorhandler(ValueError)
    def invalid_input(error):
        return jsonify({'error': str(error)}), 400

    @app.errorhandler(HTTPException)
    def http_error(error):
        if request.path.startswith('/api/'):
            return jsonify({'error': error.description}), error.code
        return error

    @app.errorhandler(Exception)
    def unexpected_error(error):
        app.logger.exception('Lỗi khi xử lý yêu cầu gợi ý phim')
        return jsonify({'error': 'Chưa thể xử lý yêu cầu. Bạn hãy thử lại sau.'}), 500

    return app


class LocalWebServer(ThreadedWSGIServer):
    """Không cho nhiều bản ứng dụng cùng chiếm một cổng trên Windows."""

    allow_reuse_address = not hasattr(socket, 'SO_EXCLUSIVEADDRUSE')

    def server_bind(self):
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def instance_id(data_dir):
    project = os.path.normcase(str(Path(__file__).resolve().parent.parent))
    dataset = os.path.normcase(str(Path(data_dir).resolve()))
    return hashlib.sha256(f'{project}\n{dataset}'.encode('utf-8')).hexdigest()[:16]


def running_instance(url, data_dir):
    """Nhận diện đúng ứng dụng đang chạy, kể cả bản cũ chưa có mã instance."""
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(f'{url}/api/health', timeout=1) as response:
            health = json.loads(response.read(65536))
            identity = response.headers.get('X-CineMatch-Instance')
        if not isinstance(health, dict) or health.get('status') != 'ready':
            return False
        if identity:
            return identity == instance_id(data_dir)
        # Compatibility with the server already running before this launcher fix.
        if Path(data_dir).resolve() != Path(DATA_DIR).resolve():
            return False
        with opener.open(url, timeout=1) as response:
            html = response.read(300000).decode('utf-8')
        return 'CineMatch' in html and 'id="recommend-form"' in html
    except (OSError, URLError, ValueError, UnicodeError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description='CineMatch — Website gợi ý phim MovieLens')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args(argv)
    url = f'http://127.0.0.1:{args.port}'
    if running_instance(url, args.data_dir):
        print(f'CineMatch đang chạy: {url}. Dùng lại máy chủ hiện có.', flush=True)
        if args.open_browser:
            webbrowser.open(url)
        return
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(('127.0.0.1', args.port)) == 0:
            parser.exit(1, f'Cổng {args.port} đang được ứng dụng khác sử dụng. '
                        'Hãy chạy UI\\run_web.bat --port 5001.\n')
    app = create_app(args.data_dir)
    # Bind successfully before reporting readiness or opening a browser tab.
    server = LocalWebServer('127.0.0.1', args.port, app)
    print(f'\nCineMatch đã sẵn sàng: {url}.', flush=True)
    if args.open_browser:
        timer = threading.Timer(1.5, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
