"""Ngăn lỗi HTML cũ đi cùng JavaScript mới và nhiều server chiếm cùng cổng."""

import contextlib
import io
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from jinja2 import FileSystemLoader

from UI.web_app import LocalWebServer, create_app


class InterfaceDeliveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.app = create_app(service=SimpleNamespace(data_dir=self.directory))
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_page_is_fresh_and_references_versioned_assets(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.cache_control.no_store)
        html = response.get_data(as_text=True)
        self.assertIn('id="theme-toggle"', html)
        self.assertEqual(re.findall(r'data-view-panel="([^"]+)"', html),
                         ['explore', 'for-you', 'comparison', 'saved'])
        assets = re.findall(r'(?:src|href)="(/static/[^\"]+)"', html)
        self.assertEqual(len(assets), 5)
        for asset in assets:
            with self.subTest(asset=asset):
                self.assertRegex(parse_qs(urlsplit(asset).query)['v'][0], r'^[a-f0-9]{12}$')
                with self.client.get(asset) as asset_response:
                    self.assertEqual(asset_response.status_code, 200)

    def use_temporary_template(self, text):
        self.app.jinja_loader = FileSystemLoader(str(self.directory))
        template = self.directory / 'index.html'
        template.write_text(text, encoding='utf-8')
        return template

    def test_template_changes_are_visible_without_restarting(self):
        template = self.use_temporary_template('First interface')
        self.assertIn(b'First interface', self.client.get('/').data)
        old_time = template.stat().st_mtime
        template.write_text('Updated interface with theme switch', encoding='utf-8')
        os.utime(template, (old_time + 2, old_time + 2))
        self.assertIn(b'Updated interface with theme switch', self.client.get('/').data)

    def test_asset_url_changes_when_its_contents_change(self):
        self.app.static_folder = str(self.directory / 'static')
        Path(self.app.static_folder).mkdir()
        script = Path(self.app.static_folder) / 'app.js'
        self.use_temporary_template("{{ url_for('static', filename='app.js') }}")
        script.write_text('oldNavigation()', encoding='utf-8')
        first = self.client.get('/').get_data(as_text=True)
        script.write_text('updatedNavigation()', encoding='utf-8')
        second = self.client.get('/').get_data(as_text=True)
        self.assertNotEqual(first, second)
        with self.client.get(second) as response:
            self.assertEqual(response.data, b'updatedNavigation()')

    def test_second_server_cannot_listen_on_the_same_port(self):
        first = LocalWebServer('127.0.0.1', 0, self.app)
        self.addCleanup(first.server_close)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                LocalWebServer('127.0.0.1', first.server_port, self.app)


if __name__ == '__main__':
    unittest.main()
