"""Kiểm tra ánh xạ poster cục bộ; không gọi API bên ngoài."""

import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from UI.posters import PosterCatalog
from UI.web_app import create_app


class PosterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def write_links(self, nested=False):
        directory = self.directory / 'ml-32m' if nested else self.directory
        directory.mkdir(exist_ok=True)
        path = directory / 'links.csv'
        path.write_text('movieId,imdbId,tmdbId\n1,0114709,862\n2571,0133093,603\n', encoding='utf-8')
        return path

    def test_plain_and_nested_links_preserve_imdb_zeros(self):
        for nested in (False, True):
            with self.subTest(nested=nested):
                path = self.write_links(nested)
                posters = PosterCatalog(self.directory)
                self.assertEqual(posters.poster_url(1), 'https://images.metahub.space/poster/medium/tt0114709/img')
                self.assertEqual(posters.poster_url(2571), 'https://images.metahub.space/poster/medium/tt0133093/img')
                self.assertIsNone(posters.poster_url(999))
                path.unlink()

    def test_zip_only_links_without_extraction(self):
        with zipfile.ZipFile(self.directory / 'ml-32m.zip', 'w') as archive:
            archive.writestr('ml-32m/links.csv', 'movieId,imdbId,tmdbId\n1,114709,862\n')
        self.assertEqual(PosterCatalog(self.directory).poster_url(1),
                         'https://images.metahub.space/poster/medium/tt0114709/img')
        self.assertFalse((self.directory / 'ml-32m').exists())

    def test_missing_or_invalid_mapping_has_no_poster(self):
        self.assertIsNone(PosterCatalog(self.directory).poster_url(1))
        (self.directory / 'links.csv').write_text(
            'movieId,imdbId\n1,https://example.com/image\n2,0\n3,../123\n4,tt0133093\n5,１２３４\n',
            encoding='utf-8',
        )
        posters = PosterCatalog(self.directory)
        self.assertTrue(all(posters.poster_url(movie_id) is None for movie_id in range(1, 6)))

    def test_redirect_route_is_local_and_cacheable(self):
        self.write_links()
        service = SimpleNamespace(data_dir=self.directory, catalog=pd.DataFrame(index=pd.Index([1, 2571, 3])))
        app = create_app(service=service)
        client = app.test_client()
        response = client.get('/api/movies/1/poster')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers['Location'], 'https://images.metahub.space/poster/medium/tt0114709/img')
        self.assertTrue(response.cache_control.public)
        self.assertEqual(response.cache_control.max_age, 86400)
        self.assertEqual(client.get('/api/movies/3/poster').status_code, 404)
        self.assertEqual(client.get('/api/movies/999/poster').status_code, 404)

    def test_mapping_is_cached_after_first_use(self):
        path = self.write_links()
        posters = PosterCatalog(self.directory)
        expected = posters.poster_url(1)
        path.unlink()
        self.assertEqual(posters.poster_url(1), expected)


if __name__ == '__main__':
    unittest.main()
