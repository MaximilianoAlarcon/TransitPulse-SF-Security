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
    heat: null
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

function initMap() {
  appState.map = L.map('map', { zoomControl: false }).setView([37.7749, -122.4194], 12);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(appState.map);
  L.control.zoom({ position: 'topright' }).addTo(appState.map);
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
}

function popupHtml(point) {
  const dt = point.incident_datetime ? new Date(point.incident_datetime).toLocaleString() : 'Unknown date';
  return `
    <div class="popup-card">
      <strong>${point.incident_category}</strong><br>
      <span>${point.incident_subcategory}</span><br>
      <span>${point.police_district}</span><br>
      <span>${point.resolution}</span><br>
      <small>${dt}</small>
    </div>
  `;
}

function riskColor(level) {
  if (level === 'high') return '#ff6b6b';
  if (level === 'medium') return '#f7b267';
  return '#66c7f4';
}

function renderMap(payload) {
  const caption = document.getElementById('map-caption');
  if (caption) {
    caption.textContent = `${Number(payload.point_count || 0)} geo incidents loaded`;
  }

  appState.layers.incidents.clearLayers();
  if (appState.layers.heat) {
    appState.map.removeLayer(appState.layers.heat);
    appState.layers.heat = null;
  }

  const points = Array.isArray(payload.points) ? payload.points : [];
  const heatData = [];
  const validLatLngs = [];

  points.forEach((point) => {
    const lat = Number(point.lat);
    const lon = Number(point.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;

    validLatLngs.push([lat, lon]);
    heatData.push([lat, lon, point.risk_level === 'high' ? 1 : point.risk_level === 'medium' ? 0.6 : 0.3]);

    const marker = L.circleMarker([lat, lon], {
      radius: point.risk_level === 'high' ? 7 : point.risk_level === 'medium' ? 5.5 : 4.5,
      color: riskColor(point.risk_level),
      weight: 1,
      fillColor: riskColor(point.risk_level),
      fillOpacity: 0.72
    }).bindPopup(popupHtml(point));

    appState.layers.incidents.addLayer(marker);
  });

  if (appState.filters.mapLayer === 'heat' && heatData.length > 0) {
    appState.layers.heat = L.heatLayer(heatData, {
      radius: 20,
      blur: 18,
      maxZoom: 15
    }).addTo(appState.map);

    if (appState.map.hasLayer(appState.layers.incidents)) {
      appState.map.removeLayer(appState.layers.incidents);
    }
  } else if (appState.pointsVisible && !appState.map.hasLayer(appState.layers.incidents)) {
    appState.layers.incidents.addTo(appState.map);
  }

  if (validLatLngs.length > 0) {
    const bounds = L.latLngBounds(validLatLngs);
    appState.map.fitBounds(bounds.pad(0.08));
  }

  requestMapResize();
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
      apiGet('/api/dashboard/map-points', getCommonParams()),
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
    refreshDashboard();
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
    const mapLayer = document.getElementById('map-layer');
    const riskMode = document.getElementById('risk-mode');

    if (timeWindow) timeWindow.value = '7d';
    if (mapLayer) mapLayer.value = 'markers';
    if (riskMode) riskMode.value = 'volume';

    refreshDashboard();
  });

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
  initMap();
  bindEvents();
  initCollapsibles();
  initDragHandle();
  clampMobileSheetHeight();
  await refreshDashboard();
  requestMapResize();
}

document.addEventListener('DOMContentLoaded', init);
