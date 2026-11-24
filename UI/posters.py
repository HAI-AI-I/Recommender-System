"""Đọc ID IMDb từ MovieLens để dùng API poster, không cần API key."""

import csv
import io
import re
import threading
import zipfile
from pathlib import Path


POSTER_BASE_URL = 'https://images.metahub.space/poster/medium/'


class PosterCatalog:

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self._mapping = None
        self._lock = threading.Lock()

    @staticmethod
    def _read_mapping(stream):
        result = {}
        for row in csv.DictReader(stream):
            movie_id = (row.get('movieId') or '').strip()
            imdb_id = (row.get('imdbId') or '').strip()
            if (not re.fullmatch(r'[0-9]{1,12}', movie_id)
                    or not re.fullmatch(r'[0-9]{1,12}', imdb_id)):
                continue
            if int(movie_id) <= 0 or int(imdb_id) <= 0:
                continue
            result[int(movie_id)] = 'tt' + imdb_id.zfill(7)
        return result

    def _load(self):
        for path in (self.data_dir / 'links.csv', self.data_dir / 'ml-32m' / 'links.csv'):
            if path.is_file():
                with path.open(encoding='utf-8-sig', newline='') as stream:
                    return self._read_mapping(stream)
        archive_path = self.data_dir / 'ml-32m.zip'
        if archive_path.is_file():
            with zipfile.ZipFile(archive_path) as archive:
                for name in ('ml-32m/links.csv', 'links.csv'):
                    if name in archive.namelist():
                        with archive.open(name) as source:
                            with io.TextIOWrapper(source, encoding='utf-8-sig', newline='') as stream:
                                return self._read_mapping(stream)
        return {}

    def poster_url(self, movie_id):
        if self._mapping is None:
            with self._lock:
                if self._mapping is None:
                    try:
                        self._mapping = self._load()
                    except (OSError, csv.Error, UnicodeError, zipfile.BadZipFile):
                        # Thiếu hoặc hỏng dữ liệu ảnh không ngăn chức năng gợi ý chạy.
                        self._mapping = {}
        imdb_id = self._mapping.get(movie_id)
        return f'{POSTER_BASE_URL}{imdb_id}/img' if imdb_id else None
