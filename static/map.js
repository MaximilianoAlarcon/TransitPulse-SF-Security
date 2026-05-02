const appState = {
  filters: {
    window: '7d',
    district: 'all',
    category: 'all',
    riskMode: 'volume',
    mapLayer: 'markers'
  },
  charts: {
    trend: null,
    category: null
  },
  map: null,
  layers: {
    incidents: null,
    heat: null,
    volumeForecast: null,
    riskForecast: null,
    hotspots: null
  },
  mapData: {
    points: [],
    center: null,
    pointCount: 0,
    riskMode: 'volume'
  },
  pointsVisible: true,
  isLoading: false,
  ui: {
    mobileDrag: {
      active: false,
      pointerId: null
    }
  }
};

function buildQuery(params) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      query.set(key, value);
    }
  });
  return query.toString();
}

async function apiGet(path, params = {}) {
  const qs = buildQuery(params);
  const url = qs ? `${path}?${qs}` : path;
  const response = await fetch(url);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json();
}

function getCommonParams() {
  return {
    window: appState.filters.window,
    district: appState.filters.district,
    category: appState.filters.category
  };
}

function getMapParams() {
  return {
    ...getCommonParams(),
    risk_mode: appState.filters.riskMode
  };
}

function initMap() {
  appState.map = L.map('map', { zoomControl: false }).setView([37.7749, -122.4194], 12);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(appState.map);
  L.control.zoom({ position: 'topright' }).addTo(appState.map);

  // Dedicated pane for isolated hotspot incidents. This keeps the small
  // context points above hotspot polygons and avoids the "data exists but is
  // visually hidden" problem.
  if (!appState.map.getPane('hotspotNoisePane')) {
    appState.map.createPane('hotspotNoisePane');
    appState.map.getPane('hotspotNoisePane').style.zIndex = 665;
  }

  appState.layers.incidents = L.layerGroup().addTo(appState.map);
}

function updateStatus(text, isError = false) {
  const el = document.getElementById('live-status');
  if (!el) return;
  el.textContent = text;
  el.classList.toggle('status-error', isError);
}

function formatNumber(value, digits = 0) {
  return new Intl.NumberFormat(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  }).format(Number(value || 0));
}

function updateKPIs(kpis) {
  document.getElementById('kpi-total').textContent = formatNumber(kpis.total_incidents, 0);
  document.getElementById('kpi-open').textContent = `${formatNumber(kpis.open_ratio, 1)}%`;
  document.getElementById('kpi-online').textContent = `${formatNumber(kpis.online_ratio, 1)}%`;
  document.getElementById('kpi-delay').textContent = `${formatNumber(kpis.avg_report_delay_minutes, 1)} min`;
}

function renderTrendChart(payload) {
  if (!payload || !Array.isArray(payload.series)) {
    throw new Error('Invalid trend payload');
  }

  const canvas = document.getElementById('trend-chart');
  if (!canvas) return;

  const labels = payload.series.map((item) => {
    const dt = item.bucket ? new Date(item.bucket) : null;
    if (!dt || Number.isNaN(dt.getTime())) return 'n/a';
    return payload.granularity === 'daily'
      ? dt.toLocaleDateString([], { month: 'short', day: 'numeric' })
      : dt.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit' });
  });
  const values = payload.series.map((item) => Number(item.total_incidents || 0));

  const granularityEl = document.getElementById('trend-granularity');
  if (granularityEl) {
    granularityEl.textContent = payload.granularity === 'daily' ? 'Daily series' : 'Hourly series';
  }

  if (appState.charts.trend) appState.charts.trend.destroy();
  appState.charts.trend = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Incidents',
        data: values,
        borderWidth: 2,
        tension: 0.25,
        fill: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: '#bcd0f7', maxTicksLimit: 8 },
          grid: { color: 'rgba(255,255,255,0.05)' }
        },
        y: {
          beginAtZero: true,
          ticks: { color: '#bcd0f7' },
          grid: { color: 'rgba(255,255,255,0.05)' }
        }
      }
    }
  });
}

function renderCategoryChart(payload) {
  if (!payload || !Array.isArray(payload.labels) || !Array.isArray(payload.values)) {
    throw new Error('Invalid category payload');
  }

  const canvas = document.getElementById('category-chart');
  if (!canvas) return;
  if (appState.charts.category) appState.charts.category.destroy();

  appState.charts.category = new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels: payload.labels,
      datasets: [{
        data: payload.values.map((value) => Number(value || 0)),
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      cutout: '62%',
      layout: {
        padding: 0
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color: '#d7e5ff',
            boxWidth: 10,
            boxHeight: 10,
            padding: 10,
            usePointStyle: true,
            pointStyle: 'circle',
            font: {
              size: 11
            }
          }
        }
      }
    }
  });
}

function riskBadgeClass(value) {
  if (value >= 40) return 'risk-high';
  if (value >= 20) return 'risk-medium';
  return 'risk-low';
}

function renderDistricts(items) {
  const container = document.getElementById('district-list');
  if (!container) return;
  container.innerHTML = '';

  if (!Array.isArray(items) || items.length === 0) {
    container.innerHTML = '<div class="stack-item"><span class="stack-title">No data for selected scope</span></div>';
    return;
  }

  items.forEach((item) => {
    const el = document.createElement('div');
    el.className = 'stack-item';
    el.innerHTML = `
      <div>
        <div class="stack-title">${item.police_district}</div>
        <div class="stack-subtitle">${formatNumber(item.total_incidents)} incidents · ${formatNumber(item.online_ratio, 1)}% online</div>
      </div>
      <span class="risk-badge ${riskBadgeClass(Number(item.open_ratio || 0))}">${formatNumber(item.open_ratio, 1)}% open</span>
    `;
    container.appendChild(el);
  });
}

function renderRiskSignals(items) {
  const container = document.getElementById('risk-signals');
  if (!container) return;
  container.innerHTML = '';

  if (!Array.isArray(items) || items.length === 0) {
    container.innerHTML = '<div class="signal-item"><div>No risk signals available.</div></div>';
    return;
  }

  items.forEach((item) => {
    const el = document.createElement('div');
    el.className = 'signal-item';
    el.innerHTML = `
      <span class="signal-dot signal-${item.severity || 'low'}"></span>
      <div class="signal-content">
        <div class="signal-label">${item.label || 'Signal'}</div>
        <div class="signal-value">${formatNumber(item.value, 1)}${item.suffix || ''}</div>
        <div class="small-muted">${item.description || ''}</div>
      </div>
    `;
    container.appendChild(el);
  });
}

function renderForecastSummary(summary) {
  const container = document.getElementById('forecast-summary');
  if (!container) return;

  const minBucket = summary.min_bucket ? new Date(summary.min_bucket).toLocaleString() : 'n/a';
  const maxBucket = summary.max_bucket ? new Date(summary.max_bucket).toLocaleString() : 'n/a';

  container.innerHTML = `
    <div class="roadmap-item">
      <strong>Training rows</strong>
      <span>${formatNumber(summary.rows_count)} rows in forecast_training_series.</span>
    </div>
    <div class="roadmap-item">
      <strong>Distinct series</strong>
      <span>${formatNumber(summary.series_count)} district/category sequences available for future forecasting.</span>
    </div>
    <div class="roadmap-item">
      <strong>Coverage window</strong>
      <span>${minBucket} → ${maxBucket}</span>
    </div>
  `;
}

function setSelectOptions(selectId, options, selectedValue) {
  const select = document.getElementById(selectId);
  if (!select) return;

  const current = selectedValue || 'all';
  select.innerHTML = '';

  (options || []).forEach((option) => {
    const el = document.createElement('option');
    el.value = option.value;
    el.textContent = option.label;
    if (option.value === current) el.selected = true;
    select.appendChild(el);
  });

  syncCustomSelect(select);
}


function createCustomSelect(select) {
  if (!select || select.dataset.customized === 'true') return;

  const wrapper = document.createElement('div');
  wrapper.className = 'custom-select';
  select.parentNode.insertBefore(wrapper, select);
  wrapper.appendChild(select);

  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'custom-select-button';
  button.setAttribute('aria-haspopup', 'listbox');
  button.setAttribute('aria-expanded', 'false');

  const label = document.createElement('span');
  label.className = 'custom-select-label';

  const chevron = document.createElement('span');
  chevron.className = 'custom-select-chevron';
  chevron.setAttribute('aria-hidden', 'true');

  button.appendChild(label);
  button.appendChild(chevron);

  const menu = document.createElement('ul');
  menu.className = 'custom-select-menu';
  menu.hidden = true;
  menu.setAttribute('role', 'listbox');

  wrapper.appendChild(button);
  wrapper.appendChild(menu);

  select.dataset.customized = 'true';
  select._customWrapper = wrapper;
  select._customButton = button;
  select._customLabel = label;
  select._customMenu = menu;

  button.addEventListener('click', (event) => {
    event.preventDefault();
    const isOpen = wrapper.classList.contains('is-open');
    closeAllCustomSelects(select);
    if (!isOpen) openCustomSelect(select);
  });

  button.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowDown' || event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      const isOpen = wrapper.classList.contains('is-open');
      closeAllCustomSelects(select);
      if (!isOpen) openCustomSelect(select);
    }
    if (event.key === 'Escape') {
      closeCustomSelect(select);
    }
  });

  select.addEventListener('change', () => {
    syncCustomSelect(select);
  });

  syncCustomSelect(select);
}

function syncCustomSelect(select) {
  if (!select) return;
  if (select.dataset.customized !== 'true') {
    createCustomSelect(select);
    return;
  }

  const wrapper = select._customWrapper;
  const label = select._customLabel;
  const menu = select._customMenu;
  if (!wrapper || !label || !menu) return;

  label.textContent = select.options[select.selectedIndex]?.textContent || 'Select';

  menu.innerHTML = '';
  Array.from(select.options).forEach((option, index) => {
    const item = document.createElement('li');
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'custom-select-option';
    if (option.selected) btn.classList.add('is-selected');
    btn.textContent = option.textContent;
    btn.setAttribute('role', 'option');
    btn.setAttribute('aria-selected', option.selected ? 'true' : 'false');

    btn.addEventListener('click', () => {
      select.selectedIndex = index;
      select.dispatchEvent(new Event('change', { bubbles: true }));
      closeCustomSelect(select);
      select._customButton?.focus();
    });

    item.appendChild(btn);
    menu.appendChild(item);
  });
}

function openCustomSelect(select) {
  const wrapper = select?._customWrapper;
  const menu = select?._customMenu;
  const button = select?._customButton;
  if (!wrapper || !menu || !button) return;

  wrapper.classList.add('is-open');
  menu.hidden = false;
  button.setAttribute('aria-expanded', 'true');
}

function closeCustomSelect(select) {
  const wrapper = select?._customWrapper;
  const menu = select?._customMenu;
  const button = select?._customButton;
  if (!wrapper || !menu || !button) return;

  wrapper.classList.remove('is-open');
  menu.hidden = true;
  button.setAttribute('aria-expanded', 'false');
}

function closeAllCustomSelects(exceptSelect = null) {
  document.querySelectorAll('select.app-select[data-customized="true"]').forEach((select) => {
    if (select !== exceptSelect) closeCustomSelect(select);
  });
}

function initCustomSelects() {
  document.querySelectorAll('select.app-select').forEach((select) => {
    createCustomSelect(select);
  });

  document.addEventListener('click', (event) => {
    if (!event.target.closest('.custom-select')) {
      closeAllCustomSelects();
    }
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      closeAllCustomSelects();
    }
  });
}


function getScoreColor(score) {
  if (score >= 0.7) return '#ff6b6b';
  if (score >= 0.4) return '#f7b267';
  return '#66c7f4';
}

function getMarkerRadius(score) {
  if (score >= 0.85) return 8;
  if (score >= 0.65) return 7;
  if (score >= 0.45) return 6;
  return 4.5;
}

function getHeatWeight(score) {
  return Math.max(0.15, Math.min(1, score));
}

function updateMapCaption() {
  const caption = document.getElementById('map-caption');
  if (!caption) return;
  const layerLabel = appState.filters.mapLayer === 'heat' ? 'Density view' : 'Incidents';
  const riskLabelMap = {
    volume: 'Volume',
    open: 'Open / Active',
    delay: 'Report delay'
  };
  const riskLabel = riskLabelMap[appState.filters.riskMode] || 'Volume';
  caption.textContent = `${layerLabel} · ${riskLabel} · ${formatNumber(appState.mapData.pointCount || 0)} points`;
}

function popupHtml(point) {
  const dt = point.incident_datetime 
    ? new Date(point.incident_datetime).toLocaleString() 
    : 'Unknown date';

  const isActive = (point.resolution || '').toLowerCase().includes('open');

  const statusLabel = isActive
    ? '🔴 Active incident'
    : '🟢 Resolved incident';

  const riskContext = isActive
    ? 'Active risk'
    : 'Area risk';

  return `
    <div class="popup-card">
      <strong>${point.incident_category}</strong><br>

      <span>${point.incident_subcategory}</span><br>
      <span>${point.police_district}</span><br>

      <span>${statusLabel}</span><br>

      <span>
        ${riskContext} (${point.risk_mode_label}): ${point.risk_level}
      </span><br>

      <small>${dt}</small>
    </div>
  `;
}

function riskColor(level) {
  if (typeof level === 'number') {
    return getScoreColor(level);
  }
  if (level === 'high') return '#ff6b6b';
  if (level === 'medium') return '#f7b267';
  return '#66c7f4';
}


function clearMapLayers() {
  if (appState.layers.incidents) {
    appState.layers.incidents.clearLayers();
    if (appState.map.hasLayer(appState.layers.incidents)) {
      appState.map.removeLayer(appState.layers.incidents);
    }
  }

  if (appState.layers.heat) {
    appState.map.removeLayer(appState.layers.heat);
    appState.layers.heat = null;
  }

  if (appState.layers.volumeForecast) {
    appState.map.removeLayer(appState.layers.volumeForecast);
    appState.layers.volumeForecast = null;
  }

  if (appState.layers.riskForecast) {
    appState.map.removeLayer(appState.layers.riskForecast);
    appState.layers.riskForecast = null;
  }

  if (appState.layers.hotspots) {
    appState.map.removeLayer(appState.layers.hotspots);
    appState.layers.hotspots = null;
  }
}

function volumeRiskColor(level) {
  if (level === 'high') return '#ff6b6b';
  if (level === 'medium') return '#f7b267';
  if (level === 'low') return '#7dd3a7';
  return '#1f2937';
}

function volumeRiskOpacity(level) {
  if (level === 'high') return 0.58;
  if (level === 'medium') return 0.48;
  if (level === 'low') return 0.38;
  return 0.14;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value || 0)));
}

function percentile(sortedValues, p) {
  if (!Array.isArray(sortedValues) || sortedValues.length === 0) return 0;

  const safeP = clamp01(p);
  const index = Math.round((sortedValues.length - 1) * safeP);
  return Number(sortedValues[Math.max(0, Math.min(index, sortedValues.length - 1))] || 0);
}

function buildRiskVisualScale(features = []) {
  const scores = features
    .map((feature) => Number(feature?.properties?.risk_score_max || 0))
    .filter((value) => Number.isFinite(value) && value >= 0)
    .sort((a, b) => a - b);

  if (scores.length === 0) {
    return {
      min: 0,
      max: 0,
      p10: 0,
      p25: 0,
      p50: 0,
      p75: 0,
      p90: 0,
      mode: 'empty'
    };
  }

  const min = scores[0];
  const max = scores[scores.length - 1];

  return {
    min,
    max,
    p10: percentile(scores, 0.10),
    p25: percentile(scores, 0.25),
    p50: percentile(scores, 0.50),
    p75: percentile(scores, 0.75),
    p90: percentile(scores, 0.90),
    mode: 'robust_percentile'
  };
}

function normalizedRiskRatio(score, scale = {}) {
  const numericScore = Math.max(0, Number(score || 0));
  const min = Number(scale.min ?? 0);
  const max = Number(scale.max ?? 0);
  const p10 = Number(scale.p10 ?? min);
  const p90 = Number(scale.p90 ?? max);

  if (!Number.isFinite(numericScore) || !Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
    return 0;
  }

  // Robust normalization: ignores the lowest and highest tails so one outlier
  // does not flatten the rest of the map visually.
  if (Number.isFinite(p10) && Number.isFinite(p90) && p90 > p10) {
    return clamp01((numericScore - p10) / (p90 - p10));
  }

  return clamp01((numericScore - min) / (max - min));
}

function lerp(a, b, t) {
  return a + (b - a) * clamp01(t);
}

function hexToRgb(hex) {
  const value = String(hex || '').replace('#', '');

  return {
    r: parseInt(value.slice(0, 2), 16),
    g: parseInt(value.slice(2, 4), 16),
    b: parseInt(value.slice(4, 6), 16)
  };
}

function rgbToHex(r, g, b) {
  return `#${[r, g, b].map((value) =>
    Math.round(value).toString(16).padStart(2, '0')
  ).join('')}`;
}

function interpolateColor(colorA, colorB, t) {
  const a = hexToRgb(colorA);
  const b = hexToRgb(colorB);
  const ratio = clamp01(t);

  return rgbToHex(
    lerp(a.r, b.r, ratio),
    lerp(a.g, b.g, ratio),
    lerp(a.b, b.b, ratio)
  );
}

function riskVisualIntensity(score, scale = {}) {
  return Math.round(normalizedRiskRatio(score, scale) * 100);
}

function riskProjectionColor(score, scale = {}) {
  const ratio = normalizedRiskRatio(score, scale);

  // Continuous analog gradient:
  // green → yellow → orange → dark red.
  if (ratio < 0.42) {
    return interpolateColor('#84cc16', '#facc15', ratio / 0.42);
  }

  if (ratio < 0.72) {
    return interpolateColor('#facc15', '#f97316', (ratio - 0.42) / 0.30);
  }

  return interpolateColor('#f97316', '#7f1d1d', (ratio - 0.72) / 0.28);
}

function riskProjectionOpacity(score, scale = {}) {
  const ratio = normalizedRiskRatio(score, scale);
  return Math.max(0.30, Math.min(0.78, 0.30 + ratio * 0.42));
}

function riskProjectionPopupHtml(properties = {}, visualScale = {}) {
  const categories = Array.isArray(properties.top_risk_categories) ? properties.top_risk_categories : [];
  const score = Number(properties.risk_score_max || 0);
  const intensity = riskVisualIntensity(score, visualScale);

  const topCategoriesHtml = categories.length > 0
    ? categories.map((item) => {
        return `<li>#${item.rank || '-'} ${item.incident_category}: ${formatNumber(item.risk_score, 4)}</li>`;
      }).join('')
    : '<li>No risk categories available</li>';

  return `
    <div class="popup-card">
      <strong>${properties.police_district || 'Unknown district'}</strong><br>
      <span>Relative district risk: ${(properties.risk_level || 'Low').toUpperCase()}</span><br>
      <span>Relative intensity: ${intensity}%</span><br>
      <span>Max risk score: ${formatNumber(properties.risk_score_max || 0, 4)}</span><br>
      <span>Avg risk score: ${formatNumber(properties.risk_score_avg || 0, 4)}</span><br>
      <hr>
      <strong>Top risk categories</strong>
      <ul>${topCategoriesHtml}</ul>
    </div>
  `;
}

function volumeProjectionPopupHtml(properties = {}) {
  const categories = Array.isArray(properties.top_categories) ? properties.top_categories : [];
  const topCategoriesHtml = categories.length > 0
    ? categories.map((item) => {
        const probability = Number(item.event_probability_next_hour || 0) * 100;
        return `<li>${item.incident_category}: ${formatNumber(item.predicted_incidents_next_hour, 4)} · ${formatNumber(probability, 1)}%</li>`;
      }).join('')
    : '<li>No projected activity</li>';

  return `
    <div class="popup-card">
      <strong>${properties.police_district || 'Unknown district'}</strong><br>
      <span>Projected incidents: ${formatNumber(properties.predicted_incidents_next_hour, 4)}</span><br>
      <span>Max event probability: ${formatNumber(Number(properties.event_probability_max || 0) * 100, 1)}%</span><br>
      <span>Level: ${(properties.risk_level || 'none').toUpperCase()}</span>
      <hr>
      <strong>Top categories</strong>
      <ul>${topCategoriesHtml}</ul>
    </div>
  `;
}

function getSfDateTimeParts(date = new Date()) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'America/Los_Angeles',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false
  }).formatToParts(date);

  const get = (type) => parts.find((part) => part.type === type)?.value;

  return {
    year: get('year'),
    month: get('month'),
    day: get('day'),
    hour: get('hour') === '24' ? '00' : get('hour'),
    minute: get('minute')
  };
}

function getCurrentSfDateTimeLocalValue() {
  const parts = getSfDateTimeParts(new Date());
  return `${parts.year}-${parts.month}-${parts.day}T${parts.hour}:${parts.minute}`;
}

function initializeVolumeProjectionTimeInput() {
  const input = document.getElementById('volume-projection-time');
  if (!input) return;

  if (!input.value) {
    input.value = getCurrentSfDateTimeLocalValue();
  }
}

function getVolumeProjectionTargetTimestamp() {
  const input = document.getElementById('volume-projection-time');

  if (!input || !input.value) {
    return getCurrentSfDateTimeLocalValue();
  }

  return input.value;
}

function initializeRiskProjectionTimeInput() {
  const input = document.getElementById('risk-projection-time');
  if (!input) return;

  if (!input.value) {
    input.value = getCurrentSfDateTimeLocalValue();
  }
}

function getRiskProjectionTargetTimestamp() {
  const input = document.getElementById('risk-projection-time');

  if (!input || !input.value) {
    return getCurrentSfDateTimeLocalValue();
  }

  return input.value;
}

function formatProjectionTimestampLabel(value) {
  if (!value || value === 'latest') return 'latest available forecast';

  const safeValue = value.length === 16 ? `${value}:00` : value;
  const parsed = new Date(safeValue);

  if (Number.isNaN(parsed.getTime())) {
    return `${value} SF`;
  }

  return parsed.toLocaleString([], {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  }) + ' SF';
}

function updateVolumeProjectionStatus(text, isError = false) {
  const el = document.getElementById('volume-projection-status');
  if (!el) return;
  el.textContent = text;
  el.classList.toggle('volume-projection-status-error', isError);
  el.classList.toggle('volume-projection-status-ok', !isError && text !== 'Not loaded yet.');
}

function renderVolumeProjectionSummary(payload) {
  const container = document.getElementById('volume-projection-summary');
  if (!container) return;

  const summary = payload?.summary || {};
  const total = formatNumber(summary.total_predicted_incidents_next_hour || 0, 4);
  const mapped = formatNumber(summary.mapped_districts || 0, 0);
  const rows = formatNumber(summary.prediction_rows || 0, 0);

  const statusEl = document.getElementById('volume-projection-status');
  const currentStatus = statusEl ? statusEl.textContent : 'Loaded.';

  container.innerHTML = `
    <div class="roadmap-item">
      <strong>Expected next hour</strong>
      <span>${total} projected incidents across ${mapped} mapped districts.</span>
    </div>
    <div class="roadmap-item">
      <strong>Model rows</strong>
      <span>${rows} district/category projections used for this layer.</span>
    </div>
    <div class="roadmap-item">
      <strong>Projection time</strong>
      <span>${formatProjectionTimestampLabel(payload?.filters?.target_timestamp)}</span>
    </div>
    <div class="volume-legend">
      <span class="volume-legend-item"><span class="volume-legend-swatch" style="background:#ff6b6b"></span>High</span>
      <span class="volume-legend-item"><span class="volume-legend-swatch" style="background:#f7b267"></span>Medium</span>
      <span class="volume-legend-item"><span class="volume-legend-swatch" style="background:#7dd3a7"></span>Low</span>
      <span class="volume-legend-item"><span class="volume-legend-swatch" style="background:#1f2937"></span>None</span>
    </div>

    <div class="volume-projection-controls mt-3">
      <label class="form-label" for="volume-projection-time">Projection time (SF)</label>
      <input
        id="volume-projection-time"
        class="form-control app-control"
        type="datetime-local"
        value="${payload?.filters?.target_timestamp && payload.filters.target_timestamp !== 'latest' ? payload.filters.target_timestamp.slice(0, 16) : getCurrentSfDateTimeLocalValue()}"
      >
      <div class="small-muted mt-1">All projection times are interpreted as San Francisco local time.</div>
    </div>

    <button id="load-volume-projections" class="btn btn-sm btn-outline-light w-100 mt-3" type="button">
      Reload volume projections
    </button>
    <div id="volume-projection-status" class="small-muted mt-2">${currentStatus || 'Loaded.'}</div>
  `;

  bindVolumeProjectionButton();
}

function buildCleanFeatureCollection(payload) {
  const features = Array.isArray(payload?.features) ? payload.features : [];

  const cleanFeatures = features
    .filter((feature) => {
      const geometryType = feature?.geometry?.type;
      const coordinates = feature?.geometry?.coordinates;
      return (
        feature?.type === 'Feature' &&
        (geometryType === 'Polygon' || geometryType === 'MultiPolygon') &&
        Array.isArray(coordinates)
      );
    })
    .map((feature) => ({
      type: 'Feature',
      geometry: feature.geometry,
      properties: feature.properties || {}
    }));

  return {
    type: 'FeatureCollection',
    features: cleanFeatures
  };
}

function renderVolumeForecastPolygons(payload) {
  clearMapLayers();

  const featureCollection = buildCleanFeatureCollection(payload);
  const features = featureCollection.features;

  appState.layers.volumeForecast = L.geoJSON(featureCollection, {
    renderer: L.svg(),
    style: (feature) => {
      const level = feature?.properties?.risk_level || 'none';
      return {
        color: 'rgba(255,255,255,0.72)',
        weight: 1.15,
        opacity: 0.95,
        fillColor: volumeRiskColor(level),
        fillOpacity: volumeRiskOpacity(level),
        lineJoin: 'round',
        lineCap: 'round'
      };
    },
    onEachFeature: (feature, layer) => {
      const properties = feature.properties || {};
      const projected = formatNumber(properties.predicted_incidents_next_hour || 0, 4);
      const label = `${properties.police_district || 'District'} · ${projected}`;

      layer.bindPopup(volumeProjectionPopupHtml(properties));
      layer.bindTooltip(label, {
        sticky: true,
        direction: 'top',
        opacity: 0.92
      });

      layer.on('mouseover', () => {
        layer.setStyle({
          weight: 2.2,
          color: 'rgba(255,255,255,0.95)',
          fillOpacity: Math.min(volumeRiskOpacity(properties.risk_level) + 0.12, 0.72)
        });
        layer.bringToFront();
      });

      layer.on('mouseout', () => {
        appState.layers.volumeForecast.resetStyle(layer);
      });
    }
  }).addTo(appState.map);

  if (features.length > 0) {
    const bounds = appState.layers.volumeForecast.getBounds();
    if (bounds && bounds.isValid()) {
      appState.map.fitBounds(bounds.pad(0.04));
    }
  } else {
    appState.map.setView([37.7749, -122.4194], 12);
  }

  const caption = document.getElementById('map-caption');
  if (caption) {
    const total = payload?.summary?.total_predicted_incidents_next_hour ?? 0;
    caption.textContent = `ML volume projections · ${formatNumber(total, 4)} expected incidents next hour`;
  }

  renderVolumeProjectionSummary(payload);
  requestMapResize();
}

async function loadVolumeProjections() {
  const button = document.getElementById('load-volume-projections');
  if (button) {
    button.disabled = true;
    button.textContent = 'Loading projections…';
  }

  updateStatus('Loading volume projections…');
  updateVolumeProjectionStatus('Loading forecast polygons…');

  try {
    initializeVolumeProjectionTimeInput();

    const targetTimestamp = getVolumeProjectionTargetTimestamp();

    const payload = await apiGet('/api/dashboard/ml-volume-polygons', {
      target_timestamp: targetTimestamp,
      district: 'all',
      category: 'all'
    });

    renderVolumeForecastPolygons(payload);
    updateStatus('Volume projections loaded');
    updateVolumeProjectionStatus('Forecast polygons loaded.');
  } catch (error) {
    console.error(error);
    updateStatus('Failed to load volume projections', true);
    updateVolumeProjectionStatus('Failed to load forecast polygons.', true);
  } finally {
    const refreshedButton = document.getElementById('load-volume-projections');
    if (refreshedButton) {
      refreshedButton.disabled = false;
      refreshedButton.textContent = appState.layers.volumeForecast ? 'Reload volume projections' : 'Load volume projections';
    }
  }
}

function bindVolumeProjectionButton() {
  const button = document.getElementById('load-volume-projections');
  if (!button || button.dataset.bound === 'true') return;

  button.dataset.bound = 'true';
  button.addEventListener('click', loadVolumeProjections);
}

function updateRiskProjectionStatus(text, isError = false) {
  const el = document.getElementById('risk-projection-status');
  if (!el) return;
  el.textContent = text;
  el.classList.toggle('risk-projection-status-error', isError);
  el.classList.toggle('risk-projection-status-ok', !isError && text !== 'Not loaded yet.');
}

function renderRiskProjectionSummary(payload) {
  const container = document.getElementById('risk-projection-summary');
  if (!container) return;

  const summary = payload?.summary || {};
  const maxScore = formatNumber(summary.max_risk_score || 0, 4);
  const avgScore = formatNumber(summary.avg_risk_score || 0, 4);
  const mapped = formatNumber(summary.mapped_districts || 0, 0);
  const rows = formatNumber(summary.prediction_rows || 0, 0);

  const thresholds = payload?.risk_score_thresholds || {};
  const visualScale = buildRiskVisualScale(Array.isArray(payload?.features) ? payload.features : []);
  const statusEl = document.getElementById('risk-projection-status');
  const currentStatus = statusEl ? statusEl.textContent : 'Loaded.';

  container.innerHTML = `
    <div class="roadmap-item">
      <strong>Relative district risk</strong>
      <span>Max score ${maxScore}, average score ${avgScore} across ${mapped} mapped districts.</span>
    </div>
    <div class="roadmap-item">
      <strong>Model rows</strong>
      <span>${rows} district/category risk projections used for this layer.</span>
    </div>
    <div class="roadmap-item">
      <strong>Projection time</strong>
      <span>${formatProjectionTimestampLabel(payload?.filters?.target_timestamp)}</span>
    </div>
    <div class="roadmap-item">
      <strong>Continuous visual scale</strong>
      <span>P10 ${formatNumber(visualScale.p10 || 0, 4)} · P50 ${formatNumber(visualScale.p50 || 0, 4)} · P90 ${formatNumber(visualScale.p90 || 0, 4)}</span>
    </div>
    <div class="roadmap-item">
      <strong>Risk thresholds</strong>
      <span>Medium ${formatNumber(thresholds.medium || 0, 4)} · High ${formatNumber(thresholds.high || 0, 4)} · Very High ${formatNumber(thresholds.very_high || 0, 4)}</span>
    </div>
    <div class="risk-legend">
      <span class="risk-legend-item"><span class="risk-legend-swatch" style="background:#84cc16"></span>Lower intensity</span>
      <span class="risk-legend-item"><span class="risk-legend-swatch" style="background:#facc15"></span>Mid intensity</span>
      <span class="risk-legend-item"><span class="risk-legend-swatch" style="background:#f97316"></span>Elevated intensity</span>
      <span class="risk-legend-item"><span class="risk-legend-swatch" style="background:#7f1d1d"></span>Higher intensity</span>
    </div>

    <div class="risk-projection-controls mt-3">
      <label class="form-label" for="risk-projection-time">Projection time (SF)</label>
      <input
        id="risk-projection-time"
        class="form-control app-control"
        type="datetime-local"
        value="${payload?.filters?.target_timestamp && payload.filters.target_timestamp !== 'latest' ? payload.filters.target_timestamp.slice(0, 16) : getCurrentSfDateTimeLocalValue()}"
      >
      <div class="small-muted mt-1">All projection times are interpreted as San Francisco local time.</div>
    </div>

    <button id="load-risk-projections" class="btn btn-sm btn-outline-light w-100 mt-3" type="button">
      Reload risk projections
    </button>
    <div id="risk-projection-status" class="small-muted mt-2">${currentStatus || 'Loaded.'}</div>
  `;

  bindRiskProjectionButton();
}

function renderRiskForecastPolygons(payload) {
  clearMapLayers();

  const featureCollection = buildCleanFeatureCollection(payload);
  const features = featureCollection.features;
  const maxRiskScore = Number(payload?.summary?.max_risk_score || payload?.max_risk_score || 0);
  const visualScale = buildRiskVisualScale(features);

  appState.layers.riskForecast = L.geoJSON(featureCollection, {
    renderer: L.svg(),
    style: (feature) => {
      const score = Number(feature?.properties?.risk_score_max || 0);
      return {
        color: 'rgba(255,255,255,0.78)',
        weight: 1.15,
        opacity: 0.95,
        fillColor: riskProjectionColor(score, visualScale),
        fillOpacity: riskProjectionOpacity(score, visualScale),
        lineJoin: 'round',
        lineCap: 'round'
      };
    },
    onEachFeature: (feature, layer) => {
      const properties = feature.properties || {};
      const score = formatNumber(properties.risk_score_max || 0, 4);
      const intensity = riskVisualIntensity(Number(properties.risk_score_max || 0), visualScale);
      const label = `${properties.police_district || 'District'} · ${intensity}% · ${score}`;

      layer.bindPopup(riskProjectionPopupHtml(properties, visualScale));
      layer.bindTooltip(label, {
        sticky: true,
        direction: 'top',
        opacity: 0.92
      });

      layer.on('mouseover', () => {
        const layerScore = Number(properties.risk_score_max || 0);
        layer.setStyle({
          weight: 2.2,
          color: 'rgba(255,255,255,0.96)',
          fillOpacity: Math.min(riskProjectionOpacity(layerScore, visualScale) + 0.12, 0.86)
        });
        layer.bringToFront();
      });

      layer.on('mouseout', () => {
        appState.layers.riskForecast.resetStyle(layer);
      });
    }
  }).addTo(appState.map);

  if (features.length > 0) {
    const bounds = appState.layers.riskForecast.getBounds();
    if (bounds && bounds.isValid()) {
      appState.map.fitBounds(bounds.pad(0.04));
    }
  } else {
    appState.map.setView([37.7749, -122.4194], 12);
  }

  const caption = document.getElementById('map-caption');
  if (caption) {
    caption.textContent = `ML risk projections · continuous intensity scale · max score ${formatNumber(maxRiskScore, 4)} · ${formatNumber(features.length, 0)} districts`;
  }

  renderRiskProjectionSummary(payload);
  requestMapResize();
}

async function loadRiskProjections() {
  const button = document.getElementById('load-risk-projections');
  if (button) {
    button.disabled = true;
    button.textContent = 'Loading risk projections…';
  }

  updateStatus('Loading risk projections…');
  updateRiskProjectionStatus('Loading risk polygons…');

  try {
    initializeRiskProjectionTimeInput();

    const targetTimestamp = getRiskProjectionTargetTimestamp();

    const payload = await apiGet('/api/dashboard/ml-risk-polygons', {
      target_timestamp: targetTimestamp,
      district: 'all',
      category: 'all'
    });

    renderRiskForecastPolygons(payload);
    updateStatus('Risk projections loaded');
    updateRiskProjectionStatus('Risk polygons loaded.');
  } catch (error) {
    console.error(error);
    updateStatus('Failed to load risk projections', true);
    updateRiskProjectionStatus('Failed to load risk polygons.', true);
  } finally {
    const refreshedButton = document.getElementById('load-risk-projections');
    if (refreshedButton) {
      refreshedButton.disabled = false;
      refreshedButton.textContent = appState.layers.riskForecast ? 'Reload risk projections' : 'Load risk projections';
    }
  }
}

function bindRiskProjectionButton() {
  const button = document.getElementById('load-risk-projections');
  if (!button || button.dataset.bound === 'true') return;

  button.dataset.bound = 'true';
  button.addEventListener('click', loadRiskProjections);
}


function renderMap(payload) {
  const points = Array.isArray(payload?.points) ? payload.points : [];
  appState.mapData = {
    points,
    center: payload?.center || null,
    pointCount: Number(payload?.point_count || points.length || 0),
    riskMode: payload?.risk_mode || appState.filters.riskMode
  };

  updateMapCaption();

  clearMapLayers();
  appState.layers.incidents.addTo(appState.map);

  const heatData = [];
  const validLatLngs = [];

  points.forEach((point) => {
    const lat = Number(point.lat);
    const lon = Number(point.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;

    const score = Math.max(0, Math.min(1, Number(point.risk_score ?? 0)));
    validLatLngs.push([lat, lon]);
    heatData.push([lat, lon, Number(point.heat_weight || getHeatWeight(score))]);

    const marker = L.circleMarker([lat, lon], {
      radius: Number(point.marker_radius || getMarkerRadius(score)),
      color: getScoreColor(score),
      weight: 1,
      fillColor: getScoreColor(score),
      fillOpacity: 0.72
    }).bindPopup(popupHtml(point));

    appState.layers.incidents.addLayer(marker);
  });

  if (appState.filters.mapLayer === 'heat' && heatData.length > 0) {
    appState.layers.heat = L.heatLayer(heatData, {
      radius: 22,
      blur: 20,
      maxZoom: 15
    }).addTo(appState.map);

    if (appState.map.hasLayer(appState.layers.incidents)) {
      appState.map.removeLayer(appState.layers.incidents);
    }
  } else if (appState.pointsVisible) {
    if (!appState.map.hasLayer(appState.layers.incidents)) {
      appState.layers.incidents.addTo(appState.map);
    }
  } else if (appState.map.hasLayer(appState.layers.incidents)) {
    appState.map.removeLayer(appState.layers.incidents);
  }

  if (validLatLngs.length > 0) {
    const bounds = L.latLngBounds(validLatLngs);
    appState.map.fitBounds(bounds.pad(0.08));
  } else if (payload?.center) {
    appState.map.setView([payload.center.lat, payload.center.lon], 12);
  }

  requestMapResize();
}

function updateHotspotStatus(text, isError = false) {
  const el = document.getElementById('hotspot-status');
  if (!el) return;
  el.textContent = text;
  el.classList.toggle('hotspot-status-error', isError);
  el.classList.toggle('hotspot-status-ok', !isError && text !== 'Not loaded yet.');
}

function getHotspotParams() {
  return {
    window: document.getElementById('hotspot-window')?.value || '7d',
    district: document.getElementById('hotspot-district-filter')?.value || 'all',
    category: document.getElementById('hotspot-category-filter')?.value || 'all'
  };
}

function hotspotColor(densityScore) {
  const score = Number(densityScore || 0);
  if (score >= 0.8) return '#ff6b6b';
  if (score >= 0.4) return '#f7b267';
  return '#66c7f4';
}

function hotspotPopupHtml(properties = {}) {
  const topDistricts = Array.isArray(properties.top_districts)
    ? properties.top_districts.map((item) => `<li>${item.district}: ${formatNumber(item.count)}</li>`).join('')
    : '';

  const topSubcategories = Array.isArray(properties.top_subcategories)
    ? properties.top_subcategories.map((item) => `<li>${item.subcategory}: ${formatNumber(item.count)}</li>`).join('')
    : '';

  const latest = properties.latest_incident_datetime
    ? new Date(properties.latest_incident_datetime).toLocaleString()
    : 'n/a';

  return `
    <div class="popup-card hotspot-popup">
      <strong>${properties.incident_category || 'Hotspot'}</strong><br>
      <span>${formatNumber(properties.point_count)} clustered incidents</span><br>
      <span>Density score: ${formatNumber(properties.density_score, 2)}</span><br>
      <span>Radius: ${formatNumber(properties.radius_meters, 0)}m</span><br>
      <span>Latest incident: ${latest}</span>
      ${topDistricts ? `<hr><strong>Top districts</strong><ul>${topDistricts}</ul>` : ''}
      ${topSubcategories ? `<hr><strong>Top subcategories</strong><ul>${topSubcategories}</ul>` : ''}
    </div>
  `;
}

function renderHotspotSummary(payload) {
  const container = document.getElementById('hotspot-summary');
  if (!container) return;

  const summary = payload?.summary || {};
  const filters = payload?.filters || {};

  const metricsHtml = `
    <div class="hotspot-metrics">
      <div class="roadmap-item">
        <strong>Hotspots detected</strong>
        <span>${formatNumber(summary.cluster_count)} clusters from ${formatNumber(summary.source_points)} source incidents.</span>
      </div>
      <div class="roadmap-item">
        <strong>Isolated incidents</strong>
        <span>${formatNumber(summary.noise_points)} incidents outside hotspot zones are shown as small context points.</span>
      </div>
      <div class="roadmap-item">
        <strong>Scope</strong>
        <span>${payload?.window || '7d'} · ${filters.district || 'all'} · ${filters.category || 'all'}</span>
      </div>
      <div class="hotspot-legend">
        <span class="hotspot-legend-item"><span class="hotspot-legend-swatch" style="background:#ff6b6b"></span>High density</span>
        <span class="hotspot-legend-item"><span class="hotspot-legend-swatch" style="background:#f7b267"></span>Medium</span>
        <span class="hotspot-legend-item"><span class="hotspot-legend-swatch" style="background:#66c7f4"></span>Lower</span>
        <span class="hotspot-legend-item"><span class="hotspot-legend-swatch" style="background:#cbd5e1"></span>Isolated incidents</span>
      </div>
    </div>
  `;

  const oldMetrics = container.querySelector('.hotspot-metrics');

  if (oldMetrics) {
    oldMetrics.outerHTML = metricsHtml;
  } else {
    const controls = container.querySelector('.hotspot-controls');
    if (controls) {
      controls.insertAdjacentHTML('beforebegin', metricsHtml);
    } else {
      container.insertAdjacentHTML('afterbegin', metricsHtml);
    }
  }
}

function hotspotNoisePopupHtml(properties = {}) {
  const dt = properties.incident_datetime
    ? new Date(properties.incident_datetime).toLocaleString()
    : 'Unknown date';

  return `
    <div class="popup-card hotspot-popup">
      <strong>${properties.incident_category || 'Isolated incident'}</strong><br>
      <span>${properties.incident_subcategory || 'Unknown subcategory'}</span><br>
      <span>${properties.police_district || 'Unknown district'}</span><br>
      <span>${properties.resolution || 'Unknown status'}</span><br>
      <small>${dt}</small>
    </div>
  `;
}

function buildHotspotPolygonFeatureCollection(payload) {
  const features = Array.isArray(payload?.cluster_features)
    ? payload.cluster_features
    : (Array.isArray(payload?.features) ? payload.features : []);
  const polygonFeatures = features.filter((feature) => {
    const geometryType = feature?.geometry?.type;
    return feature?.type === 'Feature' && (geometryType === 'Polygon' || geometryType === 'MultiPolygon');
  });

  return {
    type: 'FeatureCollection',
    features: polygonFeatures
  };
}

function getHotspotNoiseFeatures(payload) {
  const candidates = [];

  if (Array.isArray(payload?.noise_features)) {
    candidates.push(...payload.noise_features);
  }

  // Defensive support: if the backend ever returns isolated points inside the
  // main GeoJSON features array, render them too.
  if (Array.isArray(payload?.features)) {
    candidates.push(...payload.features.filter((feature) => feature?.properties?.is_noise === true));
  }

  const seen = new Set();
  return candidates.filter((feature) => {
    const coords = feature?.geometry?.coordinates;
    const id = feature?.id || feature?.properties?.hotspot_id || JSON.stringify(coords || []);
    const isValid = (
      feature?.type === 'Feature' &&
      feature?.geometry?.type === 'Point' &&
      Array.isArray(coords) &&
      Number.isFinite(Number(coords[0])) &&
      Number.isFinite(Number(coords[1]))
    );

    if (!isValid || seen.has(id)) return false;
    seen.add(id);
    return true;
  });
}

function renderHotspots(payload) {
  clearMapLayers();

  const featureCollection = buildHotspotPolygonFeatureCollection(payload);
  const features = featureCollection.features;
  const noiseFeatures = getHotspotNoiseFeatures(payload);
  const hotspotLayerGroup = L.featureGroup();
  const validLatLngs = [];

  const polygonLayer = L.geoJSON(featureCollection, {
    renderer: L.svg(),
    style: (feature) => {
      const properties = feature?.properties || {};
      const color = hotspotColor(properties.density_score);
      return {
        color,
        weight: 2,
        opacity: 0.95,
        fillColor: color,
        fillOpacity: 0.26,
        lineJoin: 'round',
        lineCap: 'round'
      };
    },
    onEachFeature: (feature, layer) => {
      const properties = feature.properties || {};
      const label = `${properties.incident_category || 'Hotspot'} · ${formatNumber(properties.point_count)} incidents`;

      layer.bindPopup(hotspotPopupHtml(properties));
      layer.bindTooltip(label, {
        sticky: true,
        direction: 'top',
        opacity: 0.92
      });

      layer.on('mouseover', () => {
        layer.setStyle({ weight: 3, fillOpacity: 0.40 });
        layer.bringToFront();
      });

      layer.on('mouseout', () => {
        polygonLayer.resetStyle(layer);
      });
    }
  });

  polygonLayer.addTo(hotspotLayerGroup);

  noiseFeatures.forEach((feature) => {
    const [lon, lat] = feature.geometry.coordinates.map(Number);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;

    validLatLngs.push([lat, lon]);

    const properties = feature.properties || {};
    const marker = L.circleMarker([lat, lon], {
      pane: 'hotspotNoisePane',
      radius: 5,
      color: '#f8fafc',
      weight: 1.4,
      opacity: 0.92,
      fillColor: '#cbd5e1',
      fillOpacity: 0.68
    }).bindPopup(hotspotNoisePopupHtml(properties));

    marker.bindTooltip(`${properties.incident_category || 'Incident'} · isolated`, {
      sticky: true,
      direction: 'top',
      opacity: 0.82
    });

    hotspotLayerGroup.addLayer(marker);
  });

  appState.layers.hotspots = hotspotLayerGroup.addTo(appState.map);

  // Fit the map to BOTH hotspot polygons and isolated incident markers.
  // The previous version only fitted to polygon bounds when clusters existed,
  // so isolated points could be correctly added but remain outside the viewport.
  const allBounds = hotspotLayerGroup.getBounds();
  if (allBounds && allBounds.isValid()) {
    appState.map.fitBounds(allBounds.pad(0.08));
  } else if (validLatLngs.length > 0) {
    appState.map.fitBounds(L.latLngBounds(validLatLngs).pad(0.08));
  } else if (payload?.center) {
    appState.map.setView([payload.center.lat, payload.center.lon], 12);
  }

  const caption = document.getElementById('map-caption');
  if (caption) {
    caption.textContent = `Hotspot clustering · ${formatNumber(payload?.summary?.cluster_count || 0)} clusters · ${formatNumber(payload?.summary?.noise_points || 0)} isolated · ${formatNumber(payload?.summary?.source_points || 0)} points analyzed`;
  }

  renderHotspotSummary(payload);
  requestMapResize();
}

async function refreshHotspotFilterOptions(selected = {}) {
  try {
    const windowValue =
      document.getElementById('hotspot-window')?.value ||
      selected.window ||
      '7d';

    const currentDistrict =
      document.getElementById('hotspot-district-filter')?.value ||
      selected.district ||
      'all';

    const currentCategory =
      document.getElementById('hotspot-category-filter')?.value ||
      selected.category ||
      'all';

    const payload = await apiGet('/api/dashboard/filters', {
      window: windowValue,
      district: 'all',
      category: 'all'
    });

    setSelectOptions('hotspot-district-filter', payload.districts, currentDistrict);
    setSelectOptions('hotspot-category-filter', payload.categories, currentCategory);
  } catch (error) {
    console.warn('Could not refresh hotspot filter options', error);
  }
}

async function loadHotspots() {
  const button = document.getElementById('load-hotspots');
  if (button) {
    button.disabled = true;
    button.textContent = 'Running clustering…';
  }

  updateStatus('Running hotspot clustering…');
  updateHotspotStatus('Cleaning map and running clustering…');
  clearMapLayers();

  try {
    const params = getHotspotParams();

    const payload = await apiGet('/api/dashboard/hotspots', params);

    const clusterCount = payload?.summary?.cluster_count || 0;
    const sourcePoints = payload?.summary?.source_points || 0;

    // Case 1: clusters and/or isolated points are available.
    if (sourcePoints > 0) {
      renderHotspots(payload);

      const isolatedCount = getHotspotNoiseFeatures(payload).length;
      if (clusterCount > 0) {
        updateStatus('Hotspots loaded');
        updateHotspotStatus(
          isolatedCount > 0
            ? `Hotspot layer loaded with ${isolatedCount} isolated incidents as context points.`
            : 'Hotspot layer loaded. No isolated incident coordinates were returned by the backend.'
        );
      } else {
        updateStatus('No dense hotspots detected');
        updateHotspotStatus(`No dense hotspots found. Showing ${isolatedCount || sourcePoints} isolated incidents instead.`);
      }
      return;
    }

    // Case 2: no incidents for selected scope.
    updateStatus('No data for selected filters');
    updateHotspotStatus(
      'No incidents found for the selected filters. Try a wider time window.'
    );

  } catch (error) {
    console.error(error);
    updateStatus('Failed to load hotspots', true);
    updateHotspotStatus('Failed to run hotspot clustering.', true);
  } finally {
    const refreshedButton = document.getElementById('load-hotspots');
    if (refreshedButton) {
      refreshedButton.disabled = false;
      refreshedButton.textContent = appState.layers.hotspots
        ? 'Reload hotspot clustering'
        : 'Run hotspot clustering';
    }
  }
}

function bindHotspotButton() {
  const button = document.getElementById('load-hotspots');
  if (button && button.dataset.bound !== 'true') {
    button.dataset.bound = 'true';
    button.addEventListener('click', loadHotspots);
  }

  const windowSelect = document.getElementById('hotspot-window');
  if (windowSelect && windowSelect.dataset.hotspotBound !== 'true') {
    windowSelect.dataset.hotspotBound = 'true';
    windowSelect.addEventListener('change', () => refreshHotspotFilterOptions(getHotspotParams()));
  }
}

function refreshVisualSizes() {
  if (appState.map && typeof appState.map.invalidateSize === 'function') {
    appState.map.invalidateSize();
  }

  if (appState.charts.trend) {
    appState.charts.trend.resize();
  }

  if (appState.charts.category) {
    appState.charts.category.resize();
  }
}

function requestMapResize() {
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      refreshVisualSizes();
    });
  });
}

async function refreshFilters() {
  const payload = await apiGet('/api/dashboard/filters', getCommonParams());
  setSelectOptions('district-filter', payload.districts, appState.filters.district);
  setSelectOptions('category-filter', payload.categories, appState.filters.category);

  const hotspotDistrictValue = document.getElementById('hotspot-district-filter')?.value || 'all';
  const hotspotCategoryValue = document.getElementById('hotspot-category-filter')?.value || 'all';
  setSelectOptions('hotspot-district-filter', payload.districts, hotspotDistrictValue);
  setSelectOptions('hotspot-category-filter', payload.categories, hotspotCategoryValue);
}

async function refreshDashboard() {
  if (appState.isLoading) return;
  appState.isLoading = true;
  updateStatus('Loading data…');

  try {
    await refreshFilters();

    const [overview, trend, districts, categories, riskSignals, mapData, forecastSummary] = await Promise.all([
      apiGet('/api/dashboard/overview', getCommonParams()),
      apiGet('/api/dashboard/trend', getCommonParams()),
      apiGet('/api/dashboard/district-pressure', getCommonParams()),
      apiGet('/api/dashboard/category-mix', getCommonParams()),
      apiGet('/api/dashboard/risk-signals', { ...getCommonParams(), risk_mode: appState.filters.riskMode }),
      apiGet('/api/dashboard/map-points', getMapParams()),
      apiGet('/api/dashboard/forecast-training-summary', getCommonParams())
    ]);

    updateKPIs(overview.kpis || {});
    renderTrendChart(trend);
    renderDistricts(districts.districts || []);
    renderCategoryChart(categories);
    renderRiskSignals(riskSignals.signals || []);
    renderMap(mapData || {});
    renderForecastSummary(forecastSummary || {});

    updateStatus('Live data connected');
    requestMapResize();
  } catch (error) {
    console.error(error);
    updateStatus('Failed to load data', true);
  } finally {
    appState.isLoading = false;
  }
}

function bindEvents() {
  document.getElementById('time-window')?.addEventListener('change', (e) => {
    appState.filters.window = e.target.value;
    appState.filters.district = 'all';
    appState.filters.category = 'all';
    refreshDashboard();
  });

  document.getElementById('district-filter')?.addEventListener('change', (e) => {
    appState.filters.district = e.target.value;
    refreshDashboard();
  });

  document.getElementById('category-filter')?.addEventListener('change', (e) => {
    appState.filters.category = e.target.value;
    refreshDashboard();
  });

  document.getElementById('map-layer')?.addEventListener('change', (e) => {
    appState.filters.mapLayer = e.target.value;
    renderMap({
      points: appState.mapData.points,
      center: appState.mapData.center,
      point_count: appState.mapData.pointCount,
      risk_mode: appState.filters.riskMode
    });
  });

  document.getElementById('risk-mode')?.addEventListener('change', (e) => {
    appState.filters.riskMode = e.target.value;
    refreshDashboard();
  });

  document.getElementById('reset-filters')?.addEventListener('click', () => {
    appState.filters = {
      window: '7d',
      district: 'all',
      category: 'all',
      riskMode: 'volume',
      mapLayer: 'markers'
    };

    const timeWindow = document.getElementById('time-window');
    const districtFilter = document.getElementById('district-filter');
    const categoryFilter = document.getElementById('category-filter');
    const mapLayer = document.getElementById('map-layer');
    const riskMode = document.getElementById('risk-mode');

    if (timeWindow) {
      timeWindow.value = '7d';
      syncCustomSelect(timeWindow);
    }
    if (districtFilter) {
      districtFilter.value = 'all';
      syncCustomSelect(districtFilter);
    }
    if (categoryFilter) {
      categoryFilter.value = 'all';
      syncCustomSelect(categoryFilter);
    }
    if (mapLayer) {
      mapLayer.value = 'markers';
      syncCustomSelect(mapLayer);
    }
    if (riskMode) {
      riskMode.value = 'volume';
      syncCustomSelect(riskMode);
    }

    refreshDashboard();
  });

  initializeVolumeProjectionTimeInput();
  initializeRiskProjectionTimeInput();
  bindVolumeProjectionButton();
  bindRiskProjectionButton();
  bindHotspotButton();
  document.getElementById('hotspot-window')?.addEventListener('change', () => refreshHotspotFilterOptions(getHotspotParams()));

  document.getElementById('btn-center-sf')?.addEventListener('click', () => {
    appState.map.setView([37.7749, -122.4194], 12);
    requestMapResize();
  });

  document.getElementById('btn-toggle-points')?.addEventListener('click', () => {
    appState.pointsVisible = !appState.pointsVisible;

    if (appState.pointsVisible) {
      if (!appState.map.hasLayer(appState.layers.incidents) && appState.filters.mapLayer !== 'heat') {
        appState.layers.incidents.addTo(appState.map);
      }
    } else if (appState.map.hasLayer(appState.layers.incidents)) {
      appState.map.removeLayer(appState.layers.incidents);
    }
  });

  window.addEventListener('resize', () => {
    clampMobileSheetHeight();
    requestMapResize();
  });
}

function initCollapsibles() {
  const cards = document.querySelectorAll('[data-collapsible]');

  cards.forEach((card) => {
    const btn = card.querySelector('.collapse-toggle');
    const body = card.querySelector('.collapsible-body');
    if (!btn || !body) return;

    btn.addEventListener('click', () => {
      const isCollapsed = card.classList.toggle('is-collapsed');
      btn.textContent = isCollapsed ? 'Show' : 'Hide';
      btn.setAttribute('aria-expanded', String(!isCollapsed));

      requestMapResize();
      window.setTimeout(requestMapResize, 320);
    });
  });
}

function getMobileSheetLimits() {
  const viewport = window.innerHeight;
  const minMapHeight = Math.round(viewport * 0.04);
  const maxMapHeight = Math.round(viewport * 0.94);
  return { minMapHeight, maxMapHeight };
}

function setMobileMapHeight(px) {
  const shell = document.getElementById('app-shell');
  if (!shell) return;

  const { minMapHeight, maxMapHeight } = getMobileSheetLimits();
  const clamped = Math.max(minMapHeight, Math.min(maxMapHeight, Math.round(px)));
  shell.style.setProperty('--mobile-map-height', `${clamped}px`);
  requestMapResize();
}

function clampMobileSheetHeight() {
  if (window.innerWidth > 991) return;

  const shell = document.getElementById('app-shell');
  if (!shell) return;

  const raw = getComputedStyle(shell).getPropertyValue('--mobile-map-height').trim();
  const current = Number.parseFloat(raw);

  if (Number.isFinite(current)) {
    setMobileMapHeight(current);
  }
}

function initDragHandle() {
  const handle = document.getElementById('drag-handle');
  if (!handle) return;

  const onPointerMove = (event) => {
    if (!appState.ui.mobileDrag.active) return;
    if (window.innerWidth > 991) return;
    setMobileMapHeight(event.clientY);
  };

  const stopDrag = () => {
    appState.ui.mobileDrag.active = false;
    appState.ui.mobileDrag.pointerId = null;
    document.body.style.userSelect = '';
    document.body.style.touchAction = '';

    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', stopDrag);
    window.removeEventListener('pointercancel', stopDrag);
  };

  handle.addEventListener('pointerdown', (event) => {
    if (window.innerWidth > 991) return;

    appState.ui.mobileDrag.active = true;
    appState.ui.mobileDrag.pointerId = event.pointerId;
    document.body.style.userSelect = 'none';
    document.body.style.touchAction = 'none';

    window.addEventListener('pointermove', onPointerMove, { passive: true });
    window.addEventListener('pointerup', stopDrag);
    window.addEventListener('pointercancel', stopDrag);
  });
}

async function init() {
  initCustomSelects();
  initMap();
  bindEvents();
  initCollapsibles();
  initDragHandle();
  clampMobileSheetHeight();
  await refreshDashboard();
  requestMapResize();
}

document.addEventListener('DOMContentLoaded', init);
