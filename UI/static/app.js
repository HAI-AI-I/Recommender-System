'use strict';

const $ = (selector) => document.querySelector(selector);
const icon = (name, className = '') => `<svg class="${className}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const escapeHTML = (value) => String(value ?? '').replace(/[&<>"']/g, (character) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));
const genreNames = {Action: 'Hành động', Adventure: 'Phiêu lưu', Animation: 'Hoạt hình', Children: 'Gia đình', Comedy: 'Hài hước', Crime: 'Tội phạm', Documentary: 'Tài liệu', Drama: 'Tâm lý', Fantasy: 'Kỳ ảo', 'Film-Noir': 'Phim noir', Horror: 'Kinh dị', Musical: 'Nhạc kịch', Mystery: 'Bí ẩn', Romance: 'Lãng mạn', 'Sci-Fi': 'Khoa học viễn tưởng', Thriller: 'Giật gân', War: 'Chiến tranh', Western: 'Viễn Tây', IMAX: 'IMAX'};

function restore(key) {
  try {
    const data = JSON.parse(localStorage.getItem(key) || '[]');
    if (!Array.isArray(data)) return new Map();
    return new Map(data.filter((movie) => movie && Number.isInteger(movie.movieId) && movie.movieId > 0 && typeof movie.clean_title === 'string').slice(0, key.includes('liked') ? 20 : 200).map((movie) => [movie.movieId, movie]));
  } catch { return new Map(); }
}

const state = {
  mode: 'new', busy: false, genre: '', config: null,
  currentView: null, hasRecommendations: false,
  liked: restore('cinematch-liked-v1'), saved: restore('cinematch-saved-v1'),
  movies: new Map(), recommendations: [], searchController: null, catalogController: null,
  catalogQuery: '', catalogMovies: [], catalogOffset: 0, catalogTotal: 0,
  catalogHasMore: false, catalogBusy: false, catalogError: false,
  detailController: null, detailMovieId: null, detailShowPrediction: true,
};
for (const [id, movie] of [...state.liked, ...state.saved]) state.movies.set(id, movie);

function persist(key, collection) {
  try { localStorage.setItem(key, JSON.stringify([...collection.values()])); }
  catch { toast('Trình duyệt chưa cho phép lưu. Danh sách vẫn dùng được trong phiên này.'); }
}

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: {'Content-Type': 'application/json', ...options.headers} });
  let data;
  try { data = await response.json(); }
  catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error('Không kết nối được với máy chủ gợi ý. Hãy kiểm tra cửa sổ chạy website.');
  }
  if (!response.ok) throw new Error(data.error || 'Chưa thể xử lý yêu cầu. Hãy thử lại.');
  return data;
}

function remember(movies) {
  for (const movie of movies) {
    state.movies.set(movie.movieId, {...state.movies.get(movie.movieId), ...movie});
  }
  return movies;
}

function toast(message) {
  const element = $('#toast');
  element.textContent = message;
  element.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { element.hidden = true; }, 3000);
}

function poster(movie) {
  const validId = Number.isSafeInteger(movie.movieId) && movie.movieId > 0;
  const title = escapeHTML(movie.clean_title);
  const image = validId
    ? `<img class="poster-image" src="/api/movies/${movie.movieId}/poster" width="300" height="450" loading="lazy" decoding="async" alt="Poster phim ${title}">`
    : '';
  return `<div class="movie-poster">${image}<div class="poster-fallback">${icon('film')}<strong class="poster-fallback-title">${title}</strong><span>Chưa có ảnh</span></div></div>`;
}

function setPosterImageState(image, loaded) {
  const container = image.closest('.movie-poster, .art-poster');
  if (!container) return;
  image.hidden = !loaded;
  container.classList.toggle('has-poster', loaded);
}

document.addEventListener('load', (event) => {
  if (event.target.matches?.('.poster-image')) {
    setPosterImageState(event.target, event.target.naturalWidth > 0);
  }
}, true);

document.addEventListener('error', (event) => {
  if (event.target.matches?.('.poster-image')) setPosterImageState(event.target, false);
}, true);

function initializePosterImages() {
  for (const image of document.querySelectorAll('.poster-image')) {
    if (image.complete) setPosterImageState(image, image.naturalWidth > 0);
  }
}

function movieCard(movie) {
  const id = movie.movieId;
  const saved = state.saved.has(id);
  const liked = state.liked.has(id);
  const firstGenre = (movie.genres || '').split('|')[0];
  const rating = Number.isFinite(movie.movie_avg_rating) ? movie.movie_avg_rating.toFixed(1) : '—';
  const score = Number.isFinite(movie.predicted_like_prob) ? `<span class="match-badge" title="Dự đoán xác suất yêu thích">Xác suất: ${Math.round(movie.predicted_like_prob * 100)}%</span>` : '';
  return `<article class="movie-card" data-movie="${id}">${score}<button class="card-save ${saved ? 'is-saved' : ''}" data-action="save" data-id="${id}" aria-label="${saved ? 'Bỏ lưu' : 'Lưu'} ${escapeHTML(movie.clean_title)}" aria-pressed="${saved}">${icon('bookmark')}</button><button class="poster-button" data-action="detail" data-id="${id}" aria-label="Xem thông tin ${escapeHTML(movie.clean_title)}">${poster(movie)}</button><div class="card-information"><button class="card-title" data-action="detail" data-id="${id}" title="${escapeHTML(movie.clean_title)}">${escapeHTML(movie.clean_title)}</button><div class="card-meta"><span>${escapeHTML(movie.release_year || 'Chưa rõ năm')}</span><span class="card-rating">${icon('star')}${rating}<span> / 5</span></span></div><div class="card-bottom"><span class="card-genre">${escapeHTML(genreNames[firstGenre] || firstGenre)}</span><button class="card-like ${liked ? 'is-liked' : ''}" data-action="like" data-id="${id}" aria-label="${liked ? 'Bỏ chọn' : 'Thêm vào phim yêu thích'} ${escapeHTML(movie.clean_title)}" aria-pressed="${liked}">${icon('heart')}</button></div></div></article>`;
}

function renderGrid(selector, movies, empty = 'Không tìm thấy phim phù hợp.') {
  remember(movies);
  $(selector).innerHTML = movies.length ? movies.map(movieCard).join('') : `<div class="empty-state"><strong>Chưa có kết quả</strong><p>${escapeHTML(empty)}</p></div>`;
}

function updateButtons() {
  $('#recommend-button').disabled = state.busy || !state.config || (state.mode === 'new' ? !state.liked.size : !Number.isInteger(Number($('#user-id').value)) || Number($('#user-id').value) < 1);
  $('#recommend-button').innerHTML = state.busy ? `${icon('shuffle', 'spinner')}<span>Đang tìm phim…</span>` : `${icon('spark')}<span>Nhận gợi ý</span>`;
  $('#saved-count').textContent = state.saved.size;
  $('#selection-hint').textContent = state.mode === 'new' ? (state.liked.size ? `Đã chọn ${state.liked.size}/20 phim.` : 'Chọn ít nhất 1 phim để bắt đầu.') : 'Gợi ý dựa trên lịch sử đánh giá. Loại các phim đã xem.';
  for (const button of document.querySelectorAll('[data-action="like"]')) {
    const liked = state.liked.has(Number(button.dataset.id));
    button.classList.toggle('is-liked', liked);
    button.setAttribute('aria-pressed', String(liked));
    const movie = state.movies.get(Number(button.dataset.id));
    if (movie) button.setAttribute('aria-label', `${liked ? 'Bỏ chọn' : 'Thêm vào phim yêu thích'} ${movie.clean_title}`);
  }
  for (const button of document.querySelectorAll('[data-action="save"]')) {
    const saved = state.saved.has(Number(button.dataset.id));
    button.classList.toggle('is-saved', saved);
    button.setAttribute('aria-pressed', String(saved));
    const movie = state.movies.get(Number(button.dataset.id));
    if (movie) button.setAttribute('aria-label', `${saved ? 'Bỏ lưu' : 'Lưu'} ${movie.clean_title}`);
  }
  window.updateComparisonContext?.();
}

function renderSelection() {
  $('#selected-movies').innerHTML = [...state.liked.values()].map((movie) => `<span class="selected-chip"><span>${escapeHTML(movie.clean_title)}</span><button type="button" data-action="remove-like" data-id="${movie.movieId}" aria-label="Bỏ chọn ${escapeHTML(movie.clean_title)}">${icon('close')}</button></span>`).join('');
  const starters = (state.config?.starter_movies || []).filter((movie) => !state.liked.has(movie.movieId));
  $('#starter-movies').innerHTML = starters.map((movie) => `<button class="starter-chip" type="button" data-action="add-like" data-id="${movie.movieId}">+ ${escapeHTML(movie.clean_title)}</button>`).join('');
  $('.starter-row').hidden = !starters.length;
  updateButtons();
}

function changeLike(id, action = 'toggle') {
  const movie = state.movies.get(id);
  if (!movie) return;
  if (action === 'remove' || (action === 'toggle' && state.liked.has(id))) state.liked.delete(id);
  else {
    if (state.liked.size >= 20 && !state.liked.has(id)) return toast('Chọn tối đa 20 phim. Bỏ chọn một phim để thêm phim khác.');
    state.liked.set(id, movie);
  }
  persist('cinematch-liked-v1', state.liked);
  renderSelection();
}

function renderSaved() {
  if (state.saved.size) renderGrid('#saved-grid', [...state.saved.values()].reverse());
  else $('#saved-grid').innerHTML = `<div class="empty-state">${icon('bookmark')}<strong>Chưa có phim đã lưu</strong><p>Nhấn nút lưu trên phim để thêm vào danh sách.</p><a class="primary-button" href="#catalog">Xem danh sách phim</a></div>`;
}

// Each menu item opens one panel; catalog is a shortcut inside Home.
const viewRoutes = {
  '#explore': {view: 'explore', panel: 'explore', label: 'Trang chủ', section: '#explore'},
  '#catalog': {view: 'catalog', panel: 'explore', label: 'Danh sách phim', section: '#catalog'},
  '#for-you': {view: 'for-you', panel: 'for-you', label: 'Dành cho bạn', section: '#for-you'},
  '#comparison': {view: 'comparison', panel: 'comparison', label: 'So sánh và đánh giá', section: '#comparison'},
  '#saved': {view: 'saved', panel: 'saved', label: 'Phim đã lưu', section: '#saved'},
};

if ('scrollRestoration' in history) history.scrollRestoration = 'manual';

function focusHeading(section) {
  const heading = section?.querySelector('h1, h2');
  if (!heading) return;
  heading.setAttribute('tabindex', '-1');
  heading.focus({preventScroll: true});
}

function scrollBehavior() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth';
}

function setView({focus = false} = {}) {
  if (location.hash === '#main-content' && state.currentView) return;
  const route = viewRoutes[location.hash] || viewRoutes['#explore'];
  const view = route.view;
  state.currentView = view;
  state.currentHash = location.hash;
  for (const panel of document.querySelectorAll('[data-view-panel]')) {
    panel.hidden = panel.dataset.viewPanel !== route.panel;
  }
  if (focus && $('#movie-dialog').open) $('#movie-dialog').close();
  const label = $('#current-view-label');
  if (label) label.textContent = route.label;
  for (const link of document.querySelectorAll('a[href^="#"]')) {
    const target = viewRoutes[link.getAttribute('href')];
    if (!target) continue;
    const active = link.classList.contains('nav-link')
      ? target.panel === route.panel : target.view === view;
    if (link.classList.contains('nav-link')) link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', link.classList.contains('nav-link') ? 'page' : 'location');
    else link.removeAttribute('aria-current');
  }
  closeSearch();
  const section = $(route.section);
  if (focus) focusHeading(section);
  if (view === 'catalog') {
    section.scrollIntoView({behavior: 'instant', block: 'start'});
  } else window.scrollTo({top: 0, left: 0, behavior: 'instant'});
}

document.addEventListener('click', (event) => {
  const link = event.target.closest('a[href^="#"]');
  if (!link || event.defaultPrevented || event.button > 0 || event.ctrlKey || event.metaKey
      || event.shiftKey || event.altKey) return;
  const hash = link.getAttribute('href');
  if (hash === '#main-content') {
    event.preventDefault();
    $('#main-content').focus({preventScroll: true});
    window.scrollTo({top: 0, left: 0, behavior: 'instant'});
    return;
  }
  if (!viewRoutes[hash]) return;
  event.preventDefault();
  if (location.hash !== hash) history.pushState(null, '', hash);
  setView({focus: true});
});

function setMode(mode) {
  state.mode = mode;
  for (const button of document.querySelectorAll('.mode-tab')) {
    const active = button.dataset.mode === mode;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
    button.tabIndex = active ? 0 : -1;
  }
  $('#new-panel').hidden = mode !== 'new';
  $('#existing-panel').hidden = mode !== 'existing';
  $('#user-id').disabled = mode !== 'existing';
  $('#form-error').hidden = true;
  closeSearch();
  updateButtons();
}

function closeSearch() {
  $('#search-results').hidden = true;
  $('#movie-search').setAttribute('aria-expanded', 'false');
}

let searchTimer;
$('#movie-search').addEventListener('input', () => {
  clearTimeout(searchTimer);
  state.searchController?.abort();
  const query = $('#movie-search').value.trim();
  if (!query) return closeSearch();
  $('#search-results').hidden = false;
  $('#movie-search').setAttribute('aria-expanded', 'true');
  $('#search-results').innerHTML = '<div class="search-message">Đang tìm phim…</div>';
  searchTimer = setTimeout(async () => {
    const controller = new AbortController();
    state.searchController = controller;
    try {
      const data = await api(`/api/movies?q=${encodeURIComponent(query)}&limit=8`, {signal: controller.signal});
      if (controller.signal.aborted || state.searchController !== controller) return;
      remember(data.movies);
      $('#search-results').innerHTML = data.movies.length ? data.movies.map((movie) => `<button type="button" class="search-result" data-action="add-like" data-id="${movie.movieId}">${icon('film')}<span><strong>${escapeHTML(movie.clean_title)}</strong><small>${escapeHTML(movie.release_year)} · ${escapeHTML((movie.genres || '').split('|').map((genre) => genreNames[genre] || genre).slice(0, 2).join(' · '))}</small></span><span class="search-add">${state.liked.has(movie.movieId) ? '✓' : '+'}</span></button>`).join('') : '<div class="search-message">Không tìm thấy phim. Thử tên gốc tiếng Anh hoặc từ khóa khác.</div>';
    } catch (error) { if (error.name !== 'AbortError') $('#search-results').innerHTML = `<div class="search-message">${escapeHTML(error.message)}</div>`; }
  }, 250);
});

$('#movie-search').addEventListener('keydown', (event) => {
  if (event.key === 'ArrowDown') { event.preventDefault(); $('#search-results button')?.focus(); }
  if (event.key === 'Escape') closeSearch();
  if (event.key === 'Enter') { event.preventDefault(); $('#search-results button')?.click(); }
});
$('#search-results').addEventListener('keydown', (event) => {
  const buttons = [...$('#search-results').querySelectorAll('button')];
  const current = buttons.indexOf(document.activeElement);
  if (event.key === 'ArrowDown') { event.preventDefault(); buttons[(current + 1) % buttons.length]?.focus(); }
  if (event.key === 'ArrowUp') { event.preventDefault(); if (current <= 0) $('#movie-search').focus(); else buttons[current - 1]?.focus(); }
  if (event.key === 'Escape') { closeSearch(); $('#movie-search').focus(); }
});
document.addEventListener('click', (event) => { if (!event.target.closest('.search-wrapper')) closeSearch(); });

function updateCatalogControls() {
  const button = $('#catalog-load-more');
  button.hidden = !state.catalogBusy && !state.catalogHasMore && !state.catalogError;
  button.disabled = state.catalogBusy;
  button.textContent = state.catalogBusy ? 'Đang tải phim…' : state.catalogError ? 'Thử tải lại' : 'Xem thêm phim';
  $('#catalog-grid').setAttribute('aria-busy', String(state.catalogBusy));
  if (!state.catalogBusy && !state.catalogError) {
    const format = new Intl.NumberFormat('vi-VN');
    $('#catalog-summary').textContent = state.catalogTotal
      ? `${format.format(state.catalogMovies.length)} trong ${format.format(state.catalogTotal)} phim`
      : 'Không có phim phù hợp';
  }
}

async function loadCatalog({append = false} = {}) {
  if (append && state.catalogBusy) return;
  state.catalogController?.abort();
  const controller = new AbortController();
  state.catalogController = controller;
  const grid = $('#catalog-grid');
  const hasRenderedCards = Boolean(grid.querySelector('.movie-card'));
  if (!append) {
    state.catalogMovies = [];
    state.catalogOffset = 0;
    state.catalogTotal = 0;
    state.catalogHasMore = false;
    $('#catalog-summary').textContent = 'Đang tìm phim…';
    if (!hasRenderedCards) grid.innerHTML = '<div class="loading-message">Đang tải danh sách phim…</div>';
  }
  const offset = state.catalogOffset;
  state.catalogBusy = true;
  state.catalogError = false;
  $('#catalog-status').textContent = append ? 'Đang tải thêm phim.' : 'Đang tìm phim.';
  updateCatalogControls();
  const parameters = new URLSearchParams({
    q: state.catalogQuery, genre: state.genre, limit: '12', offset: String(offset),
  });
  try {
    const data = await api(`/api/movies?${parameters}`, {signal: controller.signal});
    if (controller.signal.aborted || state.catalogController !== controller) return;
    const seen = new Set(state.catalogMovies.map((movie) => movie.movieId));
    const added = data.movies.filter((movie) => !seen.has(movie.movieId) && seen.add(movie.movieId));
    state.catalogMovies.push(...added);
    state.catalogOffset = offset + data.movies.length;
    state.catalogTotal = data.total;
    state.catalogHasMore = Boolean(data.has_more && data.movies.length);
    if (append) {
      remember(added);
      $('#catalog-grid').insertAdjacentHTML('beforeend', added.map(movieCard).join(''));
    } else {
      renderGrid('#catalog-grid', state.catalogMovies, 'Thử tên phim khác hoặc chọn một thể loại khác.');
    }
    $('#catalog-status').textContent = state.catalogMovies.length
      ? `${append ? `Đã tải thêm ${added.length} phim. ` : ''}Đang hiển thị ${state.catalogMovies.length} trong ${state.catalogTotal} phim.`
      : 'Không tìm thấy phim. Thử tên phim hoặc thể loại khác.';
  } catch (error) {
    if (controller.signal.aborted || state.catalogController !== controller) return;
    state.catalogError = true;
    $('#catalog-status').textContent = `${error.message} Nhấn “Thử tải lại” để tiếp tục.`;
    if (!append) {
      $('#catalog-summary').textContent = 'Chưa tải được danh sách';
      if (!hasRenderedCards) grid.innerHTML = `<div class="empty-state"><strong>Chưa tải được phim</strong><p>${escapeHTML(error.message)}</p></div>`;
    }
  } finally {
    if (state.catalogController === controller) {
      state.catalogBusy = false;
      updateCatalogControls();
    }
  }
}

let catalogTimer;
$('#catalog-search').addEventListener('input', () => {
  clearTimeout(catalogTimer);
  state.catalogController?.abort();
  state.catalogController = null;
  state.catalogQuery = $('#catalog-search').value.trim();
  state.catalogBusy = true;
  $('#catalog-summary').textContent = 'Đang tìm phim…';
  $('#catalog-status').textContent = 'Đang tìm phim.';
  updateCatalogControls();
  catalogTimer = setTimeout(() => loadCatalog(), 250);
});
$('#catalog-search').addEventListener('keydown', (event) => {
  if (event.key !== 'Enter') return;
  event.preventDefault();
  clearTimeout(catalogTimer);
  loadCatalog();
});
$('#catalog-load-more').addEventListener('click', () => loadCatalog({append: state.catalogMovies.length > 0}));

$('#genre-filters').addEventListener('click', (event) => {
  const button = event.target.closest('[data-genre]');
  if (!button || button.dataset.genre === state.genre) return;
  clearTimeout(catalogTimer);
  state.genre = button.dataset.genre;
  for (const chip of document.querySelectorAll('.genre-chip')) {
    const active = chip.dataset.genre === state.genre;
    chip.classList.toggle('active', active);
    chip.setAttribute('aria-pressed', String(active));
  }
  loadCatalog();
});

$('#recommend-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (state.busy) return;
  state.busy = true;
  $('#form-error').hidden = true;
  const status = $('#recommendation-status');
  const viewButton = $('#view-recommendations');
  if (status) status.textContent = 'Đang tạo gợi ý…';
  if (viewButton) viewButton.disabled = true;
  updateButtons();
  const payload = {mode: state.mode, top_k: Number($('#top-k').value)};
  if (state.mode === 'new') payload.liked_movie_ids = [...state.liked.keys()];
  else payload.user_id = Number($('#user-id').value);
  try {
    const data = await api('/api/recommend', {method: 'POST', body: JSON.stringify(payload)});
    state.recommendations = data.movies;
    state.hasRecommendations = true;
    renderGrid('#recommendation-grid', data.movies);
    $('#recommendation-context').textContent = data.context.mode === 'new' ? `Gợi ý dựa trên ${data.context.liked_count} phim đã chọn.` : `Dựa trên lịch sử của người dùng #${data.context.user_id}. Những phim đã xem đã được loại khỏi gợi ý.`;
    $('#history-details').hidden = !data.history.length;
    $('#history-list').innerHTML = data.history.map((movie) => `<span class="history-item">${escapeHTML(movie.clean_title)} · ${escapeHTML(movie.rating)} ★</span>`).join('');
    $('#history-details').open = false;
    $('#export-button').disabled = !data.movies.length;
    $('#recommendation-section').hidden = false;
    if (status) status.textContent = `Đã có ${data.movies.length} phim gợi ý.`;
    if (viewButton) viewButton.hidden = false;
  } catch (error) {
    $('#form-error').textContent = error.message;
    $('#form-error').hidden = false;
    if (status) status.textContent = 'Chưa tạo được gợi ý mới.';
  } finally {
    state.busy = false;
    if (viewButton) viewButton.disabled = false;
    updateButtons();
  }
});

$('#view-recommendations')?.addEventListener('click', () => {
  if (!state.hasRecommendations) return;
  const heading = $('#recommendation-title');
  heading.setAttribute('tabindex', '-1');
  heading.focus({preventScroll: true});
  heading.scrollIntoView({behavior: scrollBehavior(), block: 'start'});
});

$('#random-user').addEventListener('click', async () => {
  $('#random-user').disabled = true;
  try { const data = await api('/api/users/random'); $('#user-id').value = data.user_id; updateButtons(); }
  catch (error) { toast(error.message); }
  finally { $('#random-user').disabled = false; }
});
$('#user-id').addEventListener('input', updateButtons);
$('#top-k').addEventListener('change', updateButtons);
for (const button of document.querySelectorAll('.mode-tab')) {
  button.addEventListener('click', () => setMode(button.dataset.mode));
  button.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      const nextMode = state.mode === 'new' ? 'existing' : 'new';
      setMode(nextMode);
      $(`#${nextMode}-mode`).focus();
    }
  });
}

function renderDetail(movie, status = '') {
  const focusedAction = $('#movie-dialog').contains(document.activeElement) ? document.activeElement.dataset.action : null;
  const rating = Number.isFinite(movie.movie_avg_rating) ? `${movie.movie_avg_rating.toFixed(2)} / 5` : 'Chưa có đánh giá';
  const ratingCount = Number.isFinite(movie.movie_rating_count) ? `<span class="dialog-rating-count">${new Intl.NumberFormat('vi-VN').format(movie.movie_rating_count)} lượt đánh giá</span>` : '';
  const probability = state.detailShowPrediction && Number.isFinite(movie.predicted_like_prob) ? `<p>Dự đoán yêu thích: <strong>${Math.round(movie.predicted_like_prob * 100)}%</strong></p>` : '';
  $('#dialog-content').innerHTML = `<div class="dialog-layout">${poster(movie)}<div class="dialog-copy"><span class="dialog-eyebrow">Thông tin phim</span><h2 id="dialog-title">${escapeHTML(movie.clean_title)}</h2><span class="dialog-genre">${escapeHTML(movie.release_year)} · ${escapeHTML((movie.genres || '').split('|').map((genre) => genreNames[genre] || genre).join(' · '))}</span><p>Đánh giá cộng đồng: <strong>${rating}</strong> ${ratingCount}</p>${probability}<button class="primary-button" data-action="dialog-like" data-id="${movie.movieId}">${icon('heart')}${state.liked.has(movie.movieId) ? 'Bỏ chọn phim này' : 'Thêm vào phim yêu thích'}</button><p>Chọn phim yêu thích để dùng làm đầu vào gợi ý.</p><p class="dialog-status" role="status">${escapeHTML(status)}</p></div></div>`;
  if (focusedAction === 'dialog-like') $('#dialog-content [data-action="dialog-like"]').focus({preventScroll: true});
}

async function showDetail(movie, {showPrediction = true} = {}) {
  state.detailController?.abort();
  const controller = new AbortController();
  state.detailController = controller;
  state.detailMovieId = movie.movieId;
  state.detailShowPrediction = showPrediction;
  renderDetail(movie, 'Đang cập nhật thông tin phim…');
  if (!$('#movie-dialog').open) $('#movie-dialog').showModal();
  try {
    const data = await api(`/api/movies/${movie.movieId}`, {signal: controller.signal});
    if (controller.signal.aborted || state.detailController !== controller || !$('#movie-dialog').open) return;
    const cached = state.movies.get(movie.movieId) || movie;
    const updated = {...cached, ...data.movie};
    if (Number.isFinite(cached.predicted_like_prob)) updated.predicted_like_prob = cached.predicted_like_prob;
    state.movies.set(movie.movieId, updated);
    renderDetail(updated);
  } catch (error) {
    if (controller.signal.aborted || state.detailController !== controller || !$('#movie-dialog').open) return;
    renderDetail(state.movies.get(movie.movieId) || movie, 'Chưa tải được thông tin mới. Bạn vẫn có thể chọn phim này từ thông tin đã lưu.');
  }
}

document.addEventListener('click', (event) => {
  const button = event.target.closest('[data-action]');
  if (!button) return;
  const id = Number(button.dataset.id);
  const movie = state.movies.get(id);
  if (!movie) return;
  const action = button.dataset.action;
  if (action === 'detail') showDetail(movie);
  if (action === 'like' || action === 'dialog-like') {
    changeLike(id);
    if (action === 'dialog-like') renderDetail(movie, $('#dialog-content .dialog-status')?.textContent || '');
  }
  if (action === 'remove-like') changeLike(id, 'remove');
  if (action === 'add-like') {
    changeLike(id, 'add');
    state.searchController?.abort();
    clearTimeout(searchTimer);
    $('#movie-search').value = '';
    closeSearch();
    $('#movie-search').focus({preventScroll: true});
  }
  if (action === 'save') {
    if (state.saved.has(id)) { state.saved.delete(id); toast('Đã bỏ phim khỏi danh sách xem sau.'); }
    else {
      if (state.saved.size >= 200) return toast('Lưu tối đa 200 phim. Bỏ lưu một phim để thêm phim khác.');
      state.saved.set(id, movie); toast('Đã lưu phim.');
    }
    persist('cinematch-saved-v1', state.saved);
    updateButtons();
    renderSaved();
  }
});

$('#close-dialog').addEventListener('click', () => $('#movie-dialog').close());
$('#movie-dialog').addEventListener('close', () => {
  state.detailController?.abort();
  state.detailController = null;
  state.detailMovieId = null;
  state.detailShowPrediction = true;
});
$('#movie-dialog').addEventListener('click', (event) => {
  const rectangle = $('#movie-dialog').getBoundingClientRect();
  if (event.target === $('#movie-dialog') && (event.clientX < rectangle.left || event.clientX > rectangle.right || event.clientY < rectangle.top || event.clientY > rectangle.bottom)) $('#movie-dialog').close();
});

$('#export-button').addEventListener('click', () => {
  const cell = (value) => {
    let text = String(value ?? '');
    if (/^[=+@\-\t\r]/.test(text)) text = "'" + text;
    return '"' + text.replaceAll('"', '""') + '"';
  };
  const rows = [['Tên phim', 'Năm', 'Thể loại', 'Đánh giá / 5', 'Dự đoán yêu thích']];
  for (const movie of state.recommendations) rows.push([movie.clean_title, movie.release_year, movie.genres, movie.movie_avg_rating, movie.predicted_like_prob]);
  const url = URL.createObjectURL(new Blob(['\ufeff' + rows.map((row) => row.map(cell).join(',')).join('\r\n')], {type: 'text/csv;charset=utf-8'}));
  const link = document.createElement('a');
  link.href = url; link.download = 'cinematch-phim-danh-cho-ban.csv'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

function handleLocationChange() {
  if (state.currentHash !== location.hash) setView({focus: true});
}
window.addEventListener('hashchange', handleLocationChange);
window.addEventListener('popstate', handleLocationChange);
async function initialize() {
  const topbar = $('.topbar');
  const updateHeaderHeight = () => document.documentElement.style.setProperty('--header-height', `${topbar.getBoundingClientRect().height}px`);
  updateHeaderHeight();
  new ResizeObserver(updateHeaderHeight).observe(topbar);
  initializePosterImages();
  renderSelection();
  renderSaved();
  setMode('new');
  // Navigation and saved movies must work without waiting for the model APIs.
  setView();
  const catalogPromise = loadCatalog();
  try {
    state.config = await api('/api/config');
    remember(state.config.starter_movies);
    $('#movie-stat').textContent = `${new Intl.NumberFormat('vi-VN').format(state.config.movie_count)} phim trong mô hình`;
    if (!$('#user-id').value) $('#user-id').value = state.config.default_user_id;
    renderSelection();
  } catch (error) {
    $('#global-error').textContent = `${error.message} Tải lại trang để thử lại.`;
    $('#global-error').hidden = false;
  }
  await catalogPromise;
}
initialize();
