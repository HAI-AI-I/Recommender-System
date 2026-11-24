'use strict';

(() => {
  if (!$('#compare-form') || !$('#evaluation-form')) return;

  const analysis = {
    comparing: false, comparison: null, comparisonPayload: null,
    evaluating: false, evaluation: null, evaluationConfig: null, checkingData: false,
  };
  const number = new Intl.NumberFormat('vi-VN', {maximumFractionDigits: 3});
  const percent = new Intl.NumberFormat('vi-VN', {
    style: 'percent', minimumFractionDigits: 2, maximumFractionDigits: 2,
  });
  const methodDescriptions = {
    hybrid: 'CF chọn ứng viên, sau đó LightGBM xếp hạng theo xác suất yêu thích.',
    cf: 'Xếp hạng bằng độ tương đồng từ vector CF. Điểm CF không phải xác suất.',
    popularity: 'Ưu tiên phim có nhiều lượt đánh giá. Đây là mốc so sánh không cá nhân hóa.',
  };
  const metrics = [
    {id: 'precision', label: 'Precision', atK: true},
    {id: 'recall', label: 'Recall', atK: true},
    {id: 'ndcg', label: 'NDCG', atK: true},
    {id: 'hit_rate', label: 'Hit rate', atK: true},
    {id: 'coverage', label: 'Coverage', atK: false},
  ];

  function showError(selector, message = '') {
    const element = $(selector);
    element.textContent = message;
    element.hidden = !message;
  }

  function updateComparisonContext() {
    const topK = Number($('#top-k').value);
    const description = state.mode === 'new'
      ? state.liked.size ? `So sánh từ ${state.liked.size} phim bạn đã chọn.`
        : 'Chọn ít nhất 1 phim yêu thích ở mục Dành cho bạn để bắt đầu.'
      : `So sánh từ lịch sử của ID người dùng ${$('#user-id').value || 'chưa nhập'}.`;
    $('#compare-context').textContent = `${description} Mỗi phương pháp trả tối đa ${topK} phim.`;
    $('#compare-button').disabled = analysis.comparing || !state.config;
    $('#compare-button').textContent = analysis.comparing ? 'Đang so sánh…' : 'So sánh 3 phương pháp';
  }
  window.updateComparisonContext = updateComparisonContext;

  function comparisonPayload() {
    if (!state.config) throw new Error('Mô hình chưa sẵn sàng. Hãy chờ dữ liệu tải xong hoặc tải lại trang.');
    const payload = {mode: state.mode, top_k: Number($('#top-k').value)};
    if (!Number.isInteger(payload.top_k) || payload.top_k < 1 || payload.top_k > 50) {
      throw new Error('Số phim cần nằm trong khoảng từ 1 đến 50.');
    }
    if (state.mode === 'new') {
      payload.liked_movie_ids = [...state.liked.keys()];
      if (!payload.liked_movie_ids.length) {
        throw new Error('Hãy chọn ít nhất 1 phim yêu thích ở mục Dành cho bạn trước khi so sánh.');
      }
    } else {
      payload.user_id = Number($('#user-id').value);
      if (!Number.isSafeInteger(payload.user_id) || payload.user_id < 1) {
        throw new Error('Nhập ID người dùng hợp lệ ở mục Dành cho bạn hoặc thử ID ngẫu nhiên.');
      }
    }
    return payload;
  }

  function methodScore(methodId, movie) {
    if (methodId === 'hybrid') {
      return {label: 'Xác suất yêu thích', value: movie.predicted_like_prob,
        display: Number.isFinite(movie.predicted_like_prob) ? percent.format(movie.predicted_like_prob) : 'Chưa có điểm'};
    }
    if (methodId === 'cf') {
      return {label: 'Điểm CF', value: movie.cf_score,
        display: Number.isFinite(movie.cf_score) ? number.format(movie.cf_score) : 'Chưa có điểm'};
    }
    return {label: 'Lượt đánh giá', value: movie.movie_rating_count,
      display: Number.isFinite(movie.movie_rating_count) ? number.format(movie.movie_rating_count) : 'Chưa có thống kê'};
  }

  function renderComparison(data) {
    const snapshot = data.context.mode === 'new'
      ? `Kết quả từ ${data.context.liked_count} phim đã chọn.`
      : `Kết quả cho người dùng #${data.context.user_id}.`;
    const cards = data.methods.map((method) => {
      remember(method.movies);
      const items = method.movies.map((movie, index) => {
        const score = methodScore(method.id, movie);
        const genres = (movie.genres || '').split('|').map((genre) => genreNames[genre] || genre).join(', ');
        return `<li class="comparison-item"><span class="comparison-rank" aria-hidden="true">${index + 1}</span><div class="comparison-film-information"><button class="comparison-film" type="button" data-compare-method="${escapeHTML(method.id)}" data-movie-id="${escapeHTML(movie.movieId)}">${escapeHTML(movie.clean_title)}</button><span class="comparison-film-meta">${escapeHTML(movie.release_year || 'Chưa rõ năm')}${genres ? `, ${escapeHTML(genres)}` : ''}</span><span class="comparison-score"><span>${escapeHTML(score.label)}</span><strong>${escapeHTML(score.display)}</strong></span></div></li>`;
      }).join('');
      const timing = Number.isFinite(method.elapsed_ms) ? `${number.format(method.elapsed_ms)} ms` : '';
      return `<article class="comparison-method" data-method="${escapeHTML(method.id)}"><header class="comparison-method-header"><h3>${escapeHTML(method.name)}</h3><span class="method-time">${escapeHTML(timing)}</span></header><p class="comparison-description">${escapeHTML(methodDescriptions[method.id] || '')}</p>${items ? `<ol class="comparison-list">${items}</ol>` : '<p class="comparison-empty">Không còn phim chưa xem phù hợp để đề xuất.</p>'}</article>`;
    }).join('');
    $('#comparison-results').innerHTML = `<p class="comparison-summary" role="status">${escapeHTML(snapshot)} Tối đa ${escapeHTML(data.top_k)} phim cho mỗi phương pháp. Điểm của các phương pháp có ý nghĩa khác nhau.</p>${cards}`;
  }

  $('#compare-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (analysis.comparing) return;
    showError('#compare-error');
    let payload;
    try { payload = comparisonPayload(); }
    catch (error) { showError('#compare-error', error.message); return; }
    analysis.comparing = true;
    analysis.comparison = null;
    analysis.comparisonPayload = null;
    $('#comparison-export').hidden = true;
    $('#comparison-results').setAttribute('aria-busy', 'true');
    $('#comparison-results').innerHTML = '<p class="loading-message" role="status">Đang chạy 3 phương pháp trên cùng lựa chọn phim…</p>';
    updateComparisonContext();
    try {
      const data = await api('/api/compare', {method: 'POST', body: JSON.stringify(payload)});
      analysis.comparison = data;
      analysis.comparisonPayload = payload;
      renderComparison(data);
      $('#comparison-export').hidden = false;
    } catch (error) {
      $('#comparison-results').innerHTML = '';
      showError('#compare-error', error.message);
    } finally {
      analysis.comparing = false;
      $('#comparison-results').setAttribute('aria-busy', 'false');
      updateComparisonContext();
    }
  });

  $('#comparison-results').addEventListener('click', (event) => {
    const button = event.target.closest('[data-compare-method]');
    if (!button || !analysis.comparison) return;
    const method = analysis.comparison.methods.find((item) => item.id === button.dataset.compareMethod);
    const movie = method?.movies.find((item) => item.movieId === Number(button.dataset.movieId));
    if (movie) showDetail(movie, {showPrediction: method.id === 'hybrid'});
  });

  function updateEvaluationControls() {
    const busy = analysis.evaluating || analysis.checkingData;
    $('#evaluation-button').disabled = busy;
    $('#evaluation-users').disabled = busy;
    $('#evaluation-k').disabled = busy;
    $('#evaluation-button').textContent = analysis.evaluating ? 'Đang đánh giá…'
      : analysis.checkingData ? 'Đang kiểm tra dữ liệu…'
      : analysis.evaluationConfig?.available ? 'Chạy đánh giá' : 'Kiểm tra dữ liệu test';
    $('#evaluation-export').disabled = busy;
  }

  async function loadEvaluationConfig() {
    analysis.checkingData = true;
    updateEvaluationControls();
    $('#evaluation-status').textContent = 'Đang kiểm tra tập dữ liệu test…';
    try {
      const configuration = await api('/api/evaluation/config');
      analysis.evaluationConfig = configuration;
      $('#evaluation-users').max = String(configuration.max_users);
      $('#evaluation-status').textContent = configuration.available
        ? `Đã có dữ liệu test. Có thể đánh giá tối đa ${configuration.max_users} người dùng mỗi lần.`
        : configuration.reason || 'Chưa có dữ liệu test để đánh giá. Bổ sung dữ liệu rồi kiểm tra lại.';
      return configuration.available;
    } catch (error) {
      analysis.evaluationConfig = null;
      showError('#evaluation-error', error.message);
      $('#evaluation-status').textContent = 'Chưa kiểm tra được dữ liệu. Nhấn “Kiểm tra dữ liệu test” để thử lại.';
      return false;
    } finally {
      analysis.checkingData = false;
      updateEvaluationControls();
    }
  }

  function metricCell(method, metric, topK) {
    const value = method[metric.id];
    if (!Number.isFinite(value) || value < 0 || value > 1) return '<td>—</td>';
    const label = `${metric.label}${metric.atK ? `@${topK}` : ''}, ${method.name}`;
    return `<td><div class="metric-cell"><span class="metric-value">${escapeHTML(percent.format(value))}</span><span class="metric-track" role="meter" aria-label="${escapeHTML(label)}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${value * 100}" aria-valuetext="${escapeHTML(percent.format(value))}"><span class="metric-fill" style="width:${value * 100}%"></span></span></div></td>`;
  }

  function renderEvaluation(data) {
    const headers = metrics.map((metric) => `<th scope="col">${metric.label}${metric.atK ? `@${escapeHTML(data.top_k)}` : ''}</th>`).join('');
    const rows = data.methods.map((method) => `<tr data-method="${escapeHTML(method.id)}"><th scope="row">${escapeHTML(method.name)}</th>${metrics.map((metric) => metricCell(method, metric, data.top_k)).join('')}</tr>`).join('');
    const time = Number.isFinite(data.elapsed_ms) ? ` Thời gian: ${number.format(data.elapsed_ms)} ms.` : '';
    $('#evaluation-results').innerHTML = `<p class="evaluation-summary">Đánh giá trên cùng ${escapeHTML(data.user_count)} người dùng, Top ${escapeHTML(data.top_k)}.${escapeHTML(time)}</p><div class="evaluation-table-wrap" role="region" aria-label="Bảng so sánh chỉ số đánh giá" tabindex="0"><table class="evaluation-table"><caption>Các chỉ số hiển thị theo phần trăm trên cùng tập kiểm định.</caption><thead><tr><th scope="col">Phương pháp</th>${headers}</tr></thead><tbody>${rows}</tbody></table></div><p class="evaluation-protocol"><strong>Quy trình kiểm định: </strong>${escapeHTML(data.protocol)}</p>`;
    $('#evaluation-status').textContent = `Đã đánh giá xong ${data.user_count} người dùng với Top ${data.top_k}.`;
  }

  $('#evaluation-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (analysis.evaluating || analysis.checkingData) return;
    showError('#evaluation-error');
    const payload = {top_k: Number($('#evaluation-k').value), max_users: Number($('#evaluation-users').value)};
    const maxUsers = analysis.evaluationConfig?.max_users || 100;
    if (!Number.isInteger(payload.max_users) || payload.max_users < 1 || payload.max_users > maxUsers
        || !Number.isInteger(payload.top_k) || payload.top_k < 1 || payload.top_k > 50) {
      showError('#evaluation-error', `Chọn từ 1 đến ${maxUsers} người dùng và số phim từ 1 đến 50.`);
      return;
    }
    if (!analysis.evaluationConfig?.available && !await loadEvaluationConfig()) return;
    analysis.evaluating = true;
    analysis.evaluation = null;
    $('#evaluation-export').hidden = true;
    $('#evaluation-results').innerHTML = '';
    $('#evaluation-results').setAttribute('aria-busy', 'true');
    $('#evaluation-status').textContent = `Đang đánh giá tối đa ${payload.max_users} người dùng với Top ${payload.top_k}. Hãy chờ kết quả từ tập test.`;
    updateEvaluationControls();
    try {
      const data = await api('/api/evaluate', {method: 'POST', body: JSON.stringify(payload)});
      analysis.evaluation = data;
      renderEvaluation(data);
      $('#evaluation-export').hidden = false;
    } catch (error) {
      showError('#evaluation-error', error.message);
      $('#evaluation-status').textContent = 'Đánh giá chưa hoàn tất. Bạn có thể thử lại.';
    } finally {
      analysis.evaluating = false;
      $('#evaluation-results').setAttribute('aria-busy', 'false');
      updateEvaluationControls();
    }
  });

  function downloadCsv(filename, rows) {
    const cell = (value) => {
      let text = String(value ?? '');
      if (/^\s*[=+@-]/.test(text) || /^[\t\r\n]/.test(text)) text = "'" + text;
      return '"' + text.replaceAll('"', '""') + '"';
    };
    const csv = '\ufeff' + rows.map((row) => row.map(cell).join(',')).join('\r\n');
    const url = URL.createObjectURL(new Blob([csv], {type: 'text/csv;charset=utf-8'}));
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  $('#comparison-export').addEventListener('click', () => {
    if (!analysis.comparison || !analysis.comparisonPayload) return;
    const payload = analysis.comparisonPayload;
    const rows = [['Chế độ', 'userId', 'Phim yêu thích (ID)', 'Top K', 'Phương pháp', 'Vị trí',
      'movieId', 'Tên phim', 'Năm', 'Thể loại', 'Loại điểm', 'Điểm', 'Thời gian (ms)']];
    for (const method of analysis.comparison.methods) {
      method.movies.forEach((movie, index) => {
        const score = methodScore(method.id, movie);
        rows.push([payload.mode, payload.user_id, payload.liked_movie_ids?.join('|'), payload.top_k,
          method.name, index + 1, movie.movieId, movie.clean_title, movie.release_year,
          movie.genres, score.label, score.value, method.elapsed_ms]);
      });
    }
    downloadCsv('cinematch-so-sanh-phuong-phap.csv', rows);
  });

  $('#evaluation-export').addEventListener('click', () => {
    const data = analysis.evaluation;
    if (!data) return;
    const rows = [['Phương pháp', 'Top K', 'Số người dùng', 'Precision (0-1)', 'Recall (0-1)',
      'NDCG (0-1)', 'Hit rate (0-1)', 'Coverage (0-1)', 'Thời gian (ms)', 'Seed', 'Nguồn dữ liệu', 'Quy trình kiểm định']];
    for (const method of data.methods) {
      rows.push([method.name, data.top_k, data.user_count, ...metrics.map((metric) => method[metric.id]),
        data.elapsed_ms, data.seed, data.source, data.protocol]);
    }
    downloadCsv('cinematch-danh-gia-mo-hinh.csv', rows);
  });

  updateComparisonContext();
  loadEvaluationConfig();
})();
