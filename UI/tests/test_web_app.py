"""Kiểm tra tích hợp API với artifacts thực tế; bỏ qua khi chưa có dữ liệu."""

import unittest

from UI.web_app import DATA_DIR, create_app


@unittest.skipUnless((DATA_DIR / 'lgbm_ranker.txt').is_file(), 'Cần artifacts trong dataset/')
class WebsiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config['TESTING'] = True
        cls.service = cls.app.extensions['recommendations']

    def setUp(self):
        self.client = self.app.test_client()

    def test_home_and_configuration(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Hệ Thống Gợi Ý Phim Hybrid AI', response.get_data(as_text=True))
        configuration = self.client.get('/api/config').get_json()
        self.assertEqual(configuration['user_count'], len(self.service.engine.users))
        self.assertGreater(configuration['movie_count'], 300)
        self.assertTrue(configuration['starter_movies'])

    def test_search_uses_literal_text_and_genre(self):
        response = self.client.get('/api/movies', query_string={'q': '['})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(all('[' in movie['clean_title'] for movie in response.get_json()['movies']))
        response = self.client.get('/api/movies', query_string={'genre': 'Animation', 'limit': 5})
        movies = response.get_json()['movies']
        self.assertEqual(len(movies), 5)
        self.assertTrue(all('Animation' in movie['genres'].split('|') for movie in movies))
        response = self.client.get('/api/movies', query_string={'q': 'The Matrix'})
        self.assertIn(2571, [movie['movieId'] for movie in response.get_json()['movies']])

    def test_statistics_use_only_training_history(self):
        engine = self.service.engine
        self.assertEqual(int(engine.user_stats['user_rating_count'].sum()), engine.history.nnz)
        self.assertEqual(int(engine.movies['movie_rating_count'].sum()), engine.history.nnz)
        self.assertEqual(set(engine.user_stats.index), set(engine.user_ids))

    def test_new_user_excludes_selected_films(self):
        picked = [movie['movieId'] for movie in self.client.get('/api/config').get_json()['starter_movies']]
        response = self.client.post('/api/recommend', json={
            'mode': 'new', 'liked_movie_ids': picked, 'top_k': 10,
        })
        self.assertEqual(response.status_code, 200)
        movies = response.get_json()['movies']
        ids = [movie['movieId'] for movie in movies]
        self.assertEqual(len(ids), 10)
        self.assertEqual(len(set(ids)), 10)
        self.assertFalse(set(ids).intersection(picked))
        probabilities = [movie['predicted_like_prob'] for movie in movies]
        self.assertTrue(all(0 <= value <= 1 for value in probabilities))
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))

    def test_existing_user_excludes_watched_films(self):
        user_id = int(self.service.engine.user_ids[0])
        response = self.client.post('/api/recommend', json={
            'mode': 'existing', 'user_id': user_id, 'top_k': 5,
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        watched = self.service.recommender.user_watched.get(user_id)
        self.assertEqual(len(data['movies']), 5)
        self.assertFalse(watched.intersection(movie['movieId'] for movie in data['movies']))
        self.assertTrue(data['history'])

    def test_invalid_requests_produce_clear_errors(self):
        invalid_payloads = [
            {'mode': 'new', 'liked_movie_ids': []},
            {'mode': 'new', 'liked_movie_ids': [True]},
            {'mode': 'new', 'liked_movie_ids': [999999999]},
            {'mode': 'existing', 'user_id': 999999999},
            {'mode': 'existing', 'user_id': 1.5},
            {'mode': 'new', 'liked_movie_ids': [1], 'top_k': 0},
            {'mode': 'unsupported'},
            [],
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.client.post('/api/recommend', json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertTrue(response.get_json()['error'])
        for query in ({'limit': '-1'}, {'limit': 'abc'}, {'genre': 'unknown'}, {'q': 'a' * 151}):
            self.assertEqual(self.client.get('/api/movies', query_string=query).status_code, 400)
        self.assertEqual(self.client.get('/api/missing').status_code, 404)
        self.assertEqual(self.client.get('/api/recommend').status_code, 405)

    def test_random_user_belongs_to_model(self):
        user_id = self.client.get('/api/users/random').get_json()['user_id']
        self.assertIn(user_id, self.service.recommender.user2idx)


if __name__ == '__main__':
    unittest.main()
