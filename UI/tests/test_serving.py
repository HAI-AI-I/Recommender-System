"""Kiểm tra backend bằng mô hình nhỏ, chạy được khi chưa tải MovieLens."""

import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from final import HybridMovieRecommender, build_features, save_model_artifacts, save_serving_artifacts
from UI.web_app import SavedArtifacts, create_app


class ServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        movie_ids = [11, 22, 33, 44, 55]
        movies = pd.DataFrame({
            'movieId': movie_ids, 'clean_title': ['One [special]', 'Two', 'Three', 'Four', 'Five'],
            'release_year': [2001, 2002, 2003, 2004, 2005],
            'genres': ['Action', 'Drama', 'Action|Drama', 'Drama', 'Action'],
            'genre_Action': [1, 0, 1, 0, 1],
        })
        cls.train = pd.DataFrame({
            'userId': [7] * 4 + [8] * 5,
            'movieId': movie_ids[:4] + movie_ids,
            'rating': [5., 4., 3., 2., 1., 2., 3., 4., 5.],
        })
        movie_stats = cls.train.groupby('movieId').agg(
            movie_avg_rating=('rating', 'mean'), movie_rating_count=('rating', 'count'),
        ).reset_index()
        movie_stats['movie_log_count'] = np.log1p(movie_stats['movie_rating_count'])
        user_stats = cls.train.groupby('userId').agg(
            user_avg_rating=('rating', 'mean'), user_rating_count=('rating', 'count'),
        ).reset_index()
        user_stats['user_log_count'] = np.log1p(user_stats['user_rating_count'])
        user_vectors = np.array([[1., 0.], [.5, 1.]], dtype=np.float32)
        movie_vectors = np.array([[.1, 1.], [.5, .2], [.2, .3], [.7, .2], [.4, .5]], dtype=np.float32)
        user_mapping = {7: 0, 8: 1}
        movie_mapping = {movie_id: index for index, movie_id in enumerate(movie_ids)}
        columns = ['cf_score', 'release_year', 'genre_Action', 'movie_age', 'movie_avg_rating',
                   'movie_log_count', 'user_avg_rating', 'user_log_count', 'rating_diff_from_user_avg']
        features = build_features(cls.train, user_vectors, movie_vectors, user_mapping, movie_mapping,
                                  movies, movie_stats, user_stats, cls.train['rating'].mean())
        model = lgb.LGBMClassifier(n_estimators=3, num_leaves=2, min_child_samples=1, n_jobs=1, verbosity=-1)
        model.fit(features[columns], (cls.train['rating'] >= 4).astype(int))
        cls.recommender = HybridMovieRecommender(
            user_vectors, movie_vectors, user_mapping, movie_mapping,
            {index: movie_id for movie_id, index in movie_mapping.items()},
            movies, movie_stats, user_stats, model, columns,
        )
        save_model_artifacts(cls.directory, model, columns, {})
        save_serving_artifacts(cls.directory, cls.recommender, cls.train)
        cls.app = create_app(cls.directory)
        cls.app.config['TESTING'] = True

    def setUp(self):
        self.client = self.app.test_client()

    def test_export_and_reload_preserve_train_history(self):
        service = self.app.extensions['recommendations']
        self.assertEqual(service.engine.history.shape, (2, 5))
        self.assertEqual(service.engine.history.nnz, len(self.train))
        self.assertEqual(service.engine.feature_columns, self.recommender.feature_cols)
        self.assertEqual(service.recommender.user_watched.get(7), {11, 22, 33, 44})
        self.assertFalse(list(self.directory.glob('serving-*.tmp')))

    def test_small_catalog_and_exhausted_history(self):
        first = self.client.post('/api/recommend', json={'mode': 'existing', 'user_id': 7, 'top_k': 10})
        self.assertEqual(first.status_code, 200)
        self.assertEqual([movie['movieId'] for movie in first.json['movies']], [55])
        exhausted = self.client.post('/api/recommend', json={'mode': 'existing', 'user_id': 8})
        self.assertEqual(exhausted.status_code, 200)
        self.assertEqual(exhausted.json['movies'], [])
        new = self.client.post('/api/recommend', json={'liked_movie_ids': [11, 11], 'top_k': 10})
        self.assertEqual(new.status_code, 200)
        self.assertEqual({movie['movieId'] for movie in new.json['movies']}, {22, 33, 44, 55})
        self.assertEqual(new.json['context']['liked_count'], 1)
        probabilities = [movie['predicted_like_prob'] for movie in new.json['movies']]
        self.assertTrue(all(0 <= value <= 1 for value in probabilities))
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))
        exhausted_new = self.client.post('/api/recommend', json={'liked_movie_ids': [11, 22, 33, 44, 55]})
        self.assertEqual(exhausted_new.status_code, 200)
        self.assertEqual(exhausted_new.json['movies'], [])

    def test_catalog_pagination_details_and_literal_search(self):
        first = self.client.get('/api/movies?limit=2').json
        second = self.client.get('/api/movies?limit=2&offset=2').json
        self.assertEqual(first['total'], 5)
        self.assertTrue(first['has_more'])
        self.assertFalse({m['movieId'] for m in first['movies']} & {m['movieId'] for m in second['movies']})
        last = self.client.get('/api/movies?limit=2&offset=4').json
        self.assertFalse(last['has_more'])
        self.assertEqual(len(last['movies']), 1)
        search = self.client.get('/api/movies', query_string={'q': '[', 'genre': 'Action'}).json
        self.assertEqual([m['movieId'] for m in search['movies']], [11])
        self.assertEqual(self.client.get('/api/movies/11').json['movie']['clean_title'], 'One [special]')
        self.assertEqual(self.client.get('/api/movies/999').status_code, 404)
        ids = self.client.get('/api/movies?ids=22,11,22,999').json
        self.assertEqual([m['movieId'] for m in ids['movies']], [22, 11])
        self.assertEqual(self.client.get('/api/movies?offset=999').json['movies'], [])

    def test_history_pagination_and_health(self):
        first = self.client.get('/api/users/7/history?limit=2').json
        second = self.client.get('/api/users/7/history?limit=2&offset=2').json
        self.assertEqual(first['total'], 4)
        self.assertEqual([m['rating'] for m in first['movies']], [5., 4.])
        self.assertEqual([m['rating'] for m in second['movies']], [3., 2.])
        self.assertTrue(first['has_more'])
        self.assertFalse(second['has_more'])
        self.assertEqual(self.client.get('/api/users/999/history').status_code, 404)
        self.assertEqual(self.client.get('/api/health').json,
                         {'status': 'ready', 'movie_count': 5, 'user_count': 2})

    def test_invalid_requests_return_json_errors(self):
        for url in ('/api/movies?offset=-1', '/api/movies?offset=x', '/api/movies?ids=-1',
                    '/api/users/7/history?limit=51', '/api/users/7/history?offset=-1',
                    '/api/users/7/history?offset=x'):
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 400)
                self.assertTrue(response.json['error'])
        for options, expected in (({'data': '{', 'content_type': 'application/json'}, 400),
                                  ({'data': 'hello', 'content_type': 'text/plain'}, 415),
                                  ({'json': {'payload': 'x' * 17000}}, 413)):
            with self.subTest(expected=expected):
                response = self.client.post('/api/recommend', **options)
                self.assertEqual(response.status_code, expected)
                self.assertTrue(response.json['error'])

    def test_missing_or_mismatched_artifacts_fail_without_retraining(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ('lgbm_ranker.txt', 'feature_cols.pkl'):
                shutil.copyfile(self.directory / name, directory / name)
            with self.assertRaisesRegex(FileNotFoundError, 'serving_artifacts'):
                SavedArtifacts(directory)
            save_serving_artifacts(directory, self.recommender, self.train)
            with (directory / 'lgbm_ranker.txt').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'không khớp mô hình'):
                SavedArtifacts(directory)
            shutil.copyfile(self.directory / 'lgbm_ranker.txt', directory / 'lgbm_ranker.txt')
            with (directory / 'feature_cols.pkl').open('wb') as stream:
                pickle.dump(['wrong_feature'], stream)
            with self.assertRaisesRegex(ValueError, 'Đặc trưng không khớp'):
                SavedArtifacts(directory)


if __name__ == '__main__':
    unittest.main()
