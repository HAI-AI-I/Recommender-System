"""Chạy lại launcher phải mở server sẵn có, không khởi tạo mô hình lần nữa."""

import contextlib
import io
import unittest
from unittest.mock import patch

from UI.web_app import main


class LauncherTests(unittest.TestCase):
    def test_existing_instance_opens_browser_and_returns_success(self):
        with patch('UI.web_app.running_instance', return_value=True), \
                patch('UI.web_app.create_app') as create_app, \
                patch('UI.web_app.LocalWebServer') as server, \
                patch('UI.web_app.webbrowser.open') as browser, \
                contextlib.redirect_stdout(io.StringIO()):
            main(['--port', '5000', '--open-browser'])
        create_app.assert_not_called()
        server.assert_not_called()
        browser.assert_called_once_with('http://127.0.0.1:5000')

    def test_other_application_is_not_opened_or_replaced(self):
        with patch('UI.web_app.running_instance', return_value=False), \
                patch('UI.web_app.socket.socket') as socket, \
                patch('UI.web_app.create_app') as create_app, \
                patch('UI.web_app.webbrowser.open') as browser, \
                contextlib.redirect_stderr(io.StringIO()):
            socket.return_value.__enter__.return_value.connect_ex.return_value = 0
            with self.assertRaises(SystemExit) as result:
                main(['--port', '5000', '--open-browser'])
        self.assertEqual(result.exception.code, 1)
        create_app.assert_not_called()
        browser.assert_not_called()


if __name__ == '__main__':
    unittest.main()
