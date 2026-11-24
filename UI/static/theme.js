'use strict';

(() => {
  const storageKey = 'cinematch-theme-v1';
  const root = document.documentElement;

  function applyTheme(theme) {
    const dark = theme !== 'light';
    root.dataset.theme = dark ? 'dark' : 'light';
    document.querySelector('meta[name="theme-color"]').content = dark ? '#141414' : '#f6f6f6';
    const toggle = document.querySelector('#theme-toggle');
    if (toggle) {
      toggle.setAttribute('aria-checked', String(dark));
      toggle.title = dark ? 'Chuyển sang chế độ sáng' : 'Chuyển sang chế độ tối';
    }
  }

  // Restore the choice before styles load to avoid flashing the other theme.
  let savedTheme = 'dark';
  try { savedTheme = localStorage.getItem(storageKey); } catch {}
  applyTheme(savedTheme);

  document.addEventListener('DOMContentLoaded', () => {
    applyTheme(root.dataset.theme);
    document.querySelector('#theme-toggle').addEventListener('click', () => {
      const theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
      applyTheme(theme);
      try { localStorage.setItem(storageKey, theme); } catch {}
    });
  }, {once: true});

  window.addEventListener('storage', (event) => {
    if (event.key === storageKey || event.key === null) applyTheme(event.newValue);
  });
})();
