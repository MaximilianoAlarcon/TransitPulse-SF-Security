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
  isLoading: false
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

function riskColor(level) {
  if (level === 'high') return '#ff6b6b';
  if (level === 'medium') return '#f7b267';
  return '#66c7f4';
}

function updateStatus(text, isError = false) {
  const el = document.getElementById('live-status');
  el.textContent = text;
  el.classList.toggle('status-error', isError);
}

function updateKPIs(kpis) {
  document.getElementById('kpi-total').textContent = new Intl.NumberFormat().format(kpis.total_incidents || 0);
  document.getElementById('kpi-open').textContent = `${Number(kpis.open_ratio || 0).toFixed(1)}%`;
  document.getElementById('kpi-online').textContent = `${Number(kpis.online_ratio || 0).toFixed(1)}%`;
  document.getElementById('kpi-delay').textContent = `${Number(kpis.avg_report_delay_minutes || 0).toFixed(1)} min`;
}

let trendChartInstance = null;
let categoryChartInstance = null;

function renderTrendChart(labels, values) {
    const canvas = document.getElementById("trend-chart");
    const ctx = canvas.getContext("2d");

    if (trendChartInstance) {
        trendChartInstance.destroy();
    }

    trendChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [{
                label: "Incidents",
                data: values,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

function renderCategoryChart(labels, values) {
    const canvas = document.getElementById("category-chart");
    const ctx = canvas.getContext("2d");

    if (categoryChartInstance) {
        categoryChartInstance.destroy();
    }

    categoryChartInstance = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels,
            datasets: [{
                data: values
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
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
  container.innerHTML = '';

  if (!items.length) {
    container.innerHTML = '<div class="stack-item"><span class="stack-title">No data for selected scope</span></div>';
    return;
  }

  items.forEach(item => {
    const el = document.createElement('div');
    el.className = 'stack-item';
    el.innerHTML = `
      <div>
        <div class="stack-title">${item.police_district}</div>
        <div class="stack-subtitle">${item.total_incidents} incidents · ${item.online_ratio.toFixed(1)}% online</div>
      </div>
      <span class="risk-badge ${riskBadgeClass(item.open_ratio)}">${item.open_ratio.toFixed(1)}% open</span>
    `;
    container.appendChild(el);
  });
}

function renderRiskSignals(items) {
  const container = document.getElementById('risk-signals');
  container.innerHTML = '';

  if (!items.length) {
    container.innerHTML = '<div class="signal-item"><div>No risk signals available.</div></div>';
    return;
  }

  items.forEach(item => {
    const el = document.createElement('div');
    el.className = 'signal-item';
    el.innerHTML = `
      <span class="signal-dot signal-${item.severity}"></span>
      <div class="signal-content">
        <div class="signal-label">${item.label}</div>
        <div class="signal-value">${Number(item.value).toFixed(1)}${item.suffix || ''}</div>
        <div class="small-muted">${item.description}</div>
      </div>
    `;
    container.appendChild(el);
  });
}

function renderForecastSummary(summary) {
  const container = document.getElementById('forecast-summary');
  const minBucket = summary.min_bucket ? new Date(summary.min_bucket).toLocaleString() : 'n/a';
  const maxBucket = summary.max_bucket ? new Date(summary.max_bucket).toLocaleString() : 'n/a';

  container.innerHTML = `
    <div class="roadmap-item">
      <strong>Training rows</strong>
      <span>${new Intl.NumberFormat().format(summary.rows_count || 0)} rows in forecast_training_series.</span>
    </div>
    <div class="roadmap-item">
      <strong>Distinct series</strong>
      <span>${new Intl.NumberFormat().format(summary.series_count || 0)} district/category sequences available for future forecasting.</span>
    </div>
    <div class="roadmap-item">
      <strong>Coverage window</strong>
      <span>${minBucket} → ${maxBucket}</span>
    </div>
  `;
}

function setSelectOptions(selectId, options, selectedValue) {
  const select = document.getElementById(selectId);
  const current = selectedValue || 'all';
  select.innerHTML = '';

  options.forEach(option => {
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

function renderMap(payload) {
  document.getElementById('map-caption').textContent = `${payload.point_count} geo incidents loaded`;

  appState.layers.incidents.clearLayers();
  if (appState.layers.heat) {
    appState.map.removeLayer(appState.layers.heat);
    appState.layers.heat = null;
  }

  const heatData = [];

  payload.points.forEach(point => {
    heatData.push([point.lat, point.lon, point.risk_level === 'high' ? 1 : point.risk_level === 'medium' ? 0.6 : 0.3]);

    const marker = L.circleMarker([point.lat, point.lon], {
      radius: point.risk_level === 'high' ? 7 : point.risk_level === 'medium' ? 5.5 : 4.5,
      color: riskColor(point.risk_level),
      weight: 1,
      fillColor: riskColor(point.risk_level),
      fillOpacity: 0.72
    }).bindPopup(popupHtml(point));

    appState.layers.incidents.addLayer(marker);
  });

  if (appState.filters.mapLayer === 'heat' && heatData.length) {
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

  if (payload.points.length) {
    const bounds = L.latLngBounds(payload.points.map(p => [p.lat, p.lon]));
    appState.map.fitBounds(bounds.pad(0.08));
  }
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
    renderMap(mapData);
    renderForecastSummary(forecastSummary);

    updateStatus('Live data connected');
  } catch (error) {
    console.error(error);
    updateStatus('Failed to load data', true);
  } finally {
    appState.isLoading = false;
  }
}

function bindEvents() {
  document.getElementById('time-window').addEventListener('change', (e) => {
    appState.filters.window = e.target.value;
    appState.filters.district = 'all';
    appState.filters.category = 'all';
    refreshDashboard();
  });

  document.getElementById('district-filter').addEventListener('change', (e) => {
    appState.filters.district = e.target.value;
    refreshDashboard();
  });

  document.getElementById('category-filter').addEventListener('change', (e) => {
    appState.filters.category = e.target.value;
    refreshDashboard();
  });

  document.getElementById('map-layer').addEventListener('change', (e) => {
    appState.filters.mapLayer = e.target.value;
    refreshDashboard();
  });

  document.getElementById('risk-mode').addEventListener('change', (e) => {
    appState.filters.riskMode = e.target.value;
    refreshDashboard();
  });

  document.getElementById('reset-filters').addEventListener('click', () => {
    appState.filters = { window: '7d', district: 'all', category: 'all', riskMode: 'volume', mapLayer: 'markers' };
    document.getElementById('time-window').value = '7d';
    document.getElementById('map-layer').value = 'markers';
    document.getElementById('risk-mode').value = 'volume';
    refreshDashboard();
  });

  document.getElementById('btn-center-sf').addEventListener('click', () => {
    appState.map.setView([37.7749, -122.4194], 12);
  });

  document.getElementById('btn-toggle-points').addEventListener('click', () => {
    appState.pointsVisible = !appState.pointsVisible;
    if (appState.pointsVisible) {
      if (!appState.map.hasLayer(appState.layers.incidents)) appState.layers.incidents.addTo(appState.map);
    } else if (appState.map.hasLayer(appState.layers.incidents)) {
      appState.map.removeLayer(appState.layers.incidents);
    }
  });
}

function initDragHandle() {
  const handle = document.getElementById('drag-handle');
  const shell = document.getElementById('app-shell');
  let dragging = false;

  const move = (clientY) => {
    if (window.innerWidth > 991) return;
    const topHeight = Math.max(220, Math.min(window.innerHeight - 260, clientY));
    shell.style.setProperty('--mobile-map-height', `${topHeight}px`);
    setTimeout(() => appState.map.invalidateSize(), 0);
  };

  handle.addEventListener('touchstart', () => dragging = true, { passive: true });
  handle.addEventListener('touchend', () => dragging = false, { passive: true });
  handle.addEventListener('touchmove', (e) => {
    if (!dragging) return;
    move(e.touches[0].clientY);
  }, { passive: true });

  handle.addEventListener('mousedown', () => dragging = true);
  window.addEventListener('mouseup', () => dragging = false);
  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    move(e.clientY);
  });
}

async function init() {
  initMap();
  bindEvents();
  initDragHandle();
  await refreshDashboard();
}

document.addEventListener('DOMContentLoaded', init);
