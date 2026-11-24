"""So sánh và đánh giá trên cùng mô hình nhỏ với ground truth tính tay."""

import unittest

import pandas as pd

import test_serving as fixtures


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Dùng lại cách tạo mô hình nhỏ, không chạy lặp các bài kiểm tra của fixture.
        fixtures.ServingTests.setUpClass.__func__(cls)
        cls.test_path = cls.directory / 'test_interactions.csv'

    def setUp(self):
        self.client = self.app.test_client()
        pd.DataFrame({
            'userId': [7, 7, 7, 7, 8, 999, 7],
            'movieId': [55, 55, 11, 999, 55, 22, 44],
            'target': [1, 1, 1, 1, 1, 1, 0],
            'rating': [5, 5, 5, 5, 5, 5, 2],
        }).to_csv(self.test_path, index=False)

    def test_comparison_shares_context_and_excludes_watched(self):
        payload = {'mode': 'existing', 'user_id': 7, 'top_k': 2}
        response = self.client.post('/api/compare', json=payload)
        self.assertEqual(response.status_code, 200)
        result = response.json
        self.assertEqual(result['context'], {'mode': 'existing', 'user_id': 7})
        self.assertEqual([method['id'] for method in result['methods']], ['hybrid', 'cf', 'popularity'])
        self.assertEqual(result['top_k'], 2)
        self.assertTrue(result['history'])
        for method in result['methods']:
            self.assertEqual([movie['movieId'] for movie in method['movies']], [55])
            self.assertGreaterEqual(method['elapsed_ms'], 0)
        self.assertEqual(result['methods'][0]['movies'], self.client.post('/api/recommend', json=payload).json['movies'])
        self.assertIn('predicted_like_prob', result['methods'][0]['movies'][0])
        self.assertIn('cf_score', result['methods'][1]['movies'][0])
        for method in result['methods'][1:]:
            self.assertNotIn('predicted_like_prob', method['movies'][0])

    def test_new_user_comparison_and_empty_catalog(self):
        response = self.client.post('/api/compare', json={'liked_movie_ids': [11, 11], 'top_k': 10})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['context'], {'mode': 'new', 'liked_count': 1})
        for method in response.json['methods']:
            self.assertEqual({movie['movieId'] for movie in method['movies']}, {22, 33, 44, 55})
        exhausted = self.client.post('/api/compare', json={'mode': 'existing', 'user_id': 8})
        self.assertEqual(exhausted.status_code, 200)
        self.assertTrue(all(method['movies'] == [] for method in exhausted.json['methods']))

    def test_evaluation_uses_unseen_positives_and_same_users(self):
        response = self.client.post('/api/evaluate', json={'top_k': 2, 'max_users': 20})
        self.assertEqual(response.status_code, 200)
        result = response.json
        # Chỉ 7→55 là phim thích hợp lệ, chưa xem; các dòng trùng không được đếm lại.
        self.assertEqual(result['user_count'], 1)
        self.assertEqual(result['eligible_user_count'], 1)
        self.assertEqual(result['user_ids'], [7])
        self.assertEqual(result['seed'], 42)
        self.assertEqual(result['top_k'], 2)
        self.assertEqual(result['source'], 'test_interactions.csv')
        for method in result['methods']:
            self.assertAlmostEqual(method['precision'], .5)
            self.assertAlmostEqual(method['recall'], 1.)
            self.assertAlmostEqual(method['ndcg'], 1.)
            self.assertAlmostEqual(method['hit_rate'], 1.)
            self.assertAlmostEqual(method['coverage'], .2)
        again = self.client.post('/api/evaluate', json={'top_k': 2, 'max_users': 20}).json
        self.assertEqual(result['methods'], again['methods'])
        self.assertEqual(result['user_ids'], again['user_ids'])

    def test_evaluation_accepts_rating_only_holdout(self):
        pd.DataFrame({'userId': [7, 7], 'movieId': [55, 11], 'rating': [4., 5.]}).to_csv(self.test_path, index=False)
        response = self.client.post('/api/evaluate', json={'top_k': 1, 'max_users': 1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['user_count'], 1)
        self.assertTrue(all(method['precision'] == 1 for method in response.json['methods']))

    def test_missing_or_invalid_holdout_is_explicit(self):
        self.test_path.unlink()
        configuration = self.client.get('/api/evaluation/config')
        self.assertEqual(configuration.status_code, 200)
        self.assertFalse(configuration.json['available'])
        self.assertTrue(configuration.json['reason'])
        response = self.client.post('/api/evaluate', json={})
        self.assertEqual(response.status_code, 503)
        self.assertTrue(response.json['error'])
        pd.DataFrame({'unrelated': [1]}).to_csv(self.test_path, index=False)
        self.assertEqual(self.client.post('/api/evaluate', json={}).status_code, 400)
        pd.DataFrame({'userId': [8], 'movieId': [55], 'target': [1]}).to_csv(self.test_path, index=False)
        response = self.client.post('/api/evaluate', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn('không có người dùng', response.json['error'])

    def test_comparison_and_evaluation_validate_payloads(self):
        for route, payload in (
            ('/api/compare', []), ('/api/evaluate', []),
            ('/api/compare', {'liked_movie_ids': [True]}),
            ('/api/compare', {'mode': 'existing', 'user_id': 999}),
            ('/api/compare', {'liked_movie_ids': [11], 'top_k': 51}),
            ('/api/evaluate', {'max_users': True}), ('/api/evaluate', {'max_users': 101}),
            ('/api/evaluate', {'max_users': 0}), ('/api/evaluate', {'top_k': 0}),
        ):
            with self.subTest(route=route, payload=payload):
                response = self.client.post(route, json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertTrue(response.json['error'])


if __name__ == '__main__':
    unittest.main()
