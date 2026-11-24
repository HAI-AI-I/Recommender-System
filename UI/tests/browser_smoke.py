"""Kiểm tra giao diện trên server thật. Chạy riêng, không giả lập API mô hình.

python UI/tests/browser_smoke.py --base-url http://127.0.0.1:5000
"""

import argparse
import json
import sys
import unittest
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright


class BrowserTests(unittest.TestCase):
    base_url = 'http://127.0.0.1:5000'
    browser_channel = 'msedge'
    screenshot_directory = None

    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.addClassCleanup(cls.playwright.stop)
        options = {} if cls.browser_channel == 'chromium' else {'channel': cls.browser_channel}
        cls.browser = cls.playwright.chromium.launch(headless=True, **options)
        cls.addClassCleanup(cls.browser.close)

    def setUp(self):
        self.context = self.browser.new_context(viewport={'width': 1440, 'height': 900},
                                                reduced_motion='reduce', accept_downloads=True)
        # Poster availability is external; all application APIs still use the real server.
        self.context.route('**/poster', lambda route: route.abort())
        self.page = self.context.new_page()
        self.page.set_default_timeout(15000)
        self.errors = []
        self.documents = []
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))
        self.page.on('request', lambda request: self.documents.append(request.url)
                     if request.resource_type == 'document' else None)

    def tearDown(self):
        self.context.close()
        self.assertEqual(self.errors, [], 'JavaScript errors in the actual served interface')

    def open(self, fragment='', wait_config=True):
        self.page.goto(self.base_url + '/' + fragment, wait_until='domcontentloaded')
        if wait_config:
            self.page.wait_for_function('state.config !== null')

    def menu(self, name):
        self.page.locator(f'.nav-link[href="#{name}"]').click()
        self.assert_view(name)

    def assert_view(self, name):
        visible = self.page.locator('[data-view-panel]:visible')
        self.assertEqual(visible.evaluate_all('(panels) => panels.map(p => p.dataset.viewPanel)'), [name])
        self.assertEqual(self.page.locator(f'.nav-link[href="#{name}"]').get_attribute('aria-current'), 'page')
        self.assertEqual(self.page.evaluate('scrollY'), 0)
        self.assertLessEqual(abs(self.page.locator('.topbar').bounding_box()['y']), 1)
        self.assertTrue(self.page.get_by_role('switch').is_visible())

    def screenshot(self, name):
        if self.screenshot_directory:
            self.screenshot_directory.mkdir(parents=True, exist_ok=True)
            self.page.screenshot(path=str(self.screenshot_directory / f'{name}.png'))

    def test_menu_switches_panels_without_reloading_or_scrolling_down_the_page(self):
        self.open()
        for name in ['for-you', 'comparison', 'saved', 'for-you', 'explore']:
            self.menu(name)
        self.assertEqual(len(self.documents), 1)
        self.page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
        self.assertLessEqual(abs(self.page.locator('.topbar').bounding_box()['y']), 1)
        self.menu('saved')
        self.screenshot('saved-dark')

    def test_direct_urls_reload_and_back_forward_keep_the_right_panel(self):
        self.open('#saved')
        self.assert_view('saved')
        self.page.reload(wait_until='domcontentloaded')
        self.page.wait_for_function("state.currentView === 'saved'")
        self.assert_view('saved')
        self.menu('for-you')
        self.menu('comparison')
        self.page.go_back()
        self.page.wait_for_function("state.currentView === 'for-you'")
        self.assert_view('for-you')
        self.page.go_forward()
        self.page.wait_for_function("state.currentView === 'comparison'")
        self.assert_view('comparison')

    def test_theme_and_saved_movies_survive_reload_and_keyboard_switching(self):
        self.open()
        self.page.wait_for_selector('#catalog-grid .movie-card')
        ids = self.page.locator('#catalog-grid .movie-card').evaluate_all('(cards) => cards.slice(0, 2).map(c => c.dataset.movie)')
        for movie_id in ids:
            self.page.locator(f'#catalog-grid [data-action="save"][data-id="{movie_id}"]').click()
        self.page.get_by_role('switch').click()
        self.assertEqual(self.page.locator('html').get_attribute('data-theme'), 'light')
        self.menu('saved')
        self.assertEqual(self.page.locator('#saved-grid .movie-card').count(), 2)
        self.page.reload(wait_until='domcontentloaded')
        self.page.wait_for_function("state.currentView === 'saved'")
        self.assert_view('saved')
        self.assertEqual(self.page.locator('html').get_attribute('data-theme'), 'light')
        self.assertEqual(self.page.locator('#saved-grid .movie-card').evaluate_all('(cards) => cards.map(c => c.dataset.movie)'), list(reversed(ids)))
        self.screenshot('saved-light')
        toggle = self.page.get_by_role('switch')
        toggle.focus()
        toggle.press('Space')
        self.assertEqual(toggle.get_attribute('aria-checked'), 'true')
        toggle.press('Enter')
        self.assertEqual(toggle.get_attribute('aria-checked'), 'false')
        self.page.locator('#saved-grid [data-action="save"]').first.click()
        self.page.locator('#saved-grid [data-action="save"]').first.click()
        self.assertEqual(self.page.locator('#saved-grid .movie-card').count(), 0)
        self.assertEqual(self.page.locator('#saved-count').inner_text(), '0')
        self.assert_view('saved')

    def test_search_filter_pagination_and_movie_details(self):
        self.open('#catalog')
        self.page.wait_for_function('state.catalogMovies.length === 12')
        self.page.locator('#catalog-load-more').click()
        self.page.wait_for_function('state.catalogMovies.length === 24')
        ids = self.page.locator('#catalog-grid .movie-card').evaluate_all('(cards) => cards.map(c => c.dataset.movie)')
        self.assertEqual(len(set(ids)), 24)
        self.page.locator('#catalog-search').fill('Inception')
        self.page.locator('#catalog-search').press('Enter')
        self.page.wait_for_function("!state.catalogBusy && state.catalogQuery === 'Inception'")
        titles = self.page.locator('#catalog-grid .card-title').all_text_contents()
        self.assertTrue(titles)
        self.assertTrue(all('inception' in title.lower() for title in titles))
        self.page.locator('[data-genre="Sci-Fi"]').click()
        self.page.wait_for_function("!state.catalogBusy && state.genre === 'Sci-Fi'")
        self.page.locator('#catalog-grid [data-action="detail"]').first.click()
        self.assertTrue(self.page.locator('#movie-dialog').is_visible())
        self.assertIn('inception', self.page.locator('#dialog-title').inner_text().lower())
        self.page.locator('#close-dialog').click()
        self.page.locator('#catalog-search').fill('cinematch-no-such-film-987654321')
        self.page.locator('#catalog-search').press('Enter')
        self.page.wait_for_function('!state.catalogBusy && state.catalogTotal === 0')
        self.assertTrue(self.page.locator('#catalog-grid .empty-state').is_visible())

    def test_recommendations_comparison_and_csv_use_the_real_model(self):
        self.open('#for-you')
        self.page.locator('#starter-movies button').first.click()
        self.page.locator('#top-k').select_option('5')
        self.page.locator('#recommend-button').click()
        self.page.wait_for_function('state.hasRecommendations && !state.busy')
        self.assertEqual(self.page.locator('#recommendation-grid .movie-card').count(), 5)
        with self.page.expect_download() as download:
            self.page.locator('#export-button').click()
        self.assertEqual(len(Path(download.value.path()).read_text(encoding='utf-8-sig').splitlines()), 6)
        self.menu('comparison')
        self.page.locator('#compare-button').click()
        self.page.wait_for_selector('.comparison-method')
        self.assertEqual(self.page.locator('.comparison-method').count(), 3)
        with self.page.expect_download() as download:
            self.page.locator('#comparison-export').click()
        self.assertGreater(len(Path(download.value.path()).read_text(encoding='utf-8-sig').splitlines()), 3)
        self.screenshot('comparison-dark')

    def test_evaluation_and_export_on_real_holdout(self):
        self.open('#comparison')
        self.page.wait_for_function("!document.querySelector('#evaluation-button').disabled")
        self.page.locator('#evaluation-users').fill('1')
        self.page.locator('#evaluation-k').select_option('5')
        self.page.locator('#evaluation-button').click()
        self.page.wait_for_selector('.evaluation-table')
        self.assertEqual(self.page.locator('.evaluation-table tbody tr').count(), 3)
        with self.page.expect_download() as download:
            self.page.locator('#evaluation-export').click()
        self.assertEqual(len(Path(download.value.path()).read_text(encoding='utf-8-sig').splitlines()), 4)
        self.page.get_by_role('switch').click()
        self.screenshot('evaluation-light')

    def test_saved_and_theme_remain_available_when_model_requests_fail(self):
        with urllib.request.urlopen(self.base_url + '/api/movies?limit=1', timeout=10) as response:
            movie = json.load(response)['movies'][0]
        self.context.add_init_script("localStorage.setItem('cinematch-saved-v1', " + json.dumps(json.dumps([movie])) + ');')
        self.page.route('**/api/config', lambda route: route.fulfill(status=503, json={'error': 'Model unavailable'}))
        self.page.route('**/api/movies?*', lambda route: route.fulfill(status=503, json={'error': 'Catalog unavailable'}))
        self.open('#saved', wait_config=False)
        self.page.wait_for_selector('#saved-grid .movie-card')
        self.assert_view('saved')
        self.page.get_by_role('switch').click()
        self.assertEqual(self.page.locator('html').get_attribute('data-theme'), 'light')
        self.menu('comparison')
        self.menu('saved')

    def test_mobile_and_desktop_layout_in_both_themes(self):
        self.open()
        for width, height in [(320, 568), (390, 844), (740, 800), (800, 600), (1024, 768), (1920, 1080)]:
            self.page.set_viewport_size({'width': width, 'height': height})
            for theme in ['light', 'dark']:
                if self.page.locator('html').get_attribute('data-theme') != theme:
                    self.page.get_by_role('switch').click()
                for name in ['for-you', 'comparison', 'saved', 'explore']:
                    with self.subTest(width=width, theme=theme, panel=name):
                        self.menu(name)
                        self.assertFalse(self.page.evaluate('document.documentElement.scrollWidth > innerWidth'))
                        switch = self.page.get_by_role('switch').bounding_box()
                        self.assertGreaterEqual(switch['x'], 0)
                        self.assertLessEqual(switch['x'] + switch['width'], width)
            if width in (390, 1920):
                self.screenshot(f'home-dark-{width}')
                self.page.get_by_role('switch').click()
                self.screenshot(f'home-light-{width}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Browser regression checks against a running CineMatch server')
    parser.add_argument('--base-url', default=BrowserTests.base_url)
    parser.add_argument('--browser', choices=['msedge', 'chrome', 'chromium'], default='msedge')
    parser.add_argument('--screenshots', type=Path)
    options = parser.parse_args()
    BrowserTests.base_url = options.base_url.rstrip('/')
    BrowserTests.browser_channel = options.browser
    BrowserTests.screenshot_directory = options.screenshots
    unittest.main(argv=[sys.argv[0]], verbosity=2)
