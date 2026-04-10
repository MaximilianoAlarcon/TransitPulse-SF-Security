// ===============================
// 🌍 INIT MAP (Leaflet)
// ===============================
const map = L.map("map", {
    zoomControl: false
}).setView([37.7749, -122.4194], 12); // San Francisco

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors"
}).addTo(map);

L.control.zoom({ position: "topright" }).addTo(map);

// Layer groups
const incidentsLayer = L.layerGroup().addTo(map);

// ===============================
// 📊 MOCK DATA (luego reemplazar por API)
// ===============================
const mockKPIs = {
    total: 1245,
    open: 342,
    online: 210,
    avgDelay: 18
};

const mockTrend = [120, 180, 150, 220, 300, 280, 260];

const mockCategories = {
    labels: ["Theft", "Assault", "Burglary", "Vandalism"],
    values: [45, 20, 15, 20]
};

const mockDistricts = [
    { name: "Mission", value: 320 },
    { name: "Tenderloin", value: 290 },
    { name: "SOMA", value: 250 },
    { name: "Sunset", value: 120 }
];

const mockRisk = {
    last1h: 15,
    last3h: 40,
    last24h: 280,
    last7d: 1600
};

// ===============================
// 📌 UPDATE KPIs
// ===============================
function updateKPIs() {
    document.getElementById("kpi-total").innerText = mockKPIs.total;
    document.getElementById("kpi-open").innerText = mockKPIs.open;
    document.getElementById("kpi-online").innerText = mockKPIs.online;
    document.getElementById("kpi-delay").innerText = mockKPIs.avgDelay + " min";
}

// ===============================
// 📈 TREND CHART
// ===============================
function initTrendChart() {
    new Chart(document.getElementById("trendChart"), {
        type: "line",
        data: {
            labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            datasets: [{
                label: "Incidents",
                data: mockTrend,
                tension: 0.3
            }]
        }
    });
}

// ===============================
// 📊 CATEGORY CHART
// ===============================
function initCategoryChart() {
    new Chart(document.getElementById("categoryChart"), {
        type: "doughnut",
        data: {
            labels: mockCategories.labels,
            datasets: [{
                data: mockCategories.values
            }]
        }
    });
}

// ===============================
// 🏙️ DISTRICT LIST
// ===============================
function renderDistricts() {
    const container = document.getElementById("district-list");
    container.innerHTML = "";

    mockDistricts.forEach(d => {
        const el = document.createElement("div");
        el.className = "district-item";
        el.innerHTML = `<span>${d.name}</span><strong>${d.value}</strong>`;
        container.appendChild(el);
    });
}

// ===============================
// ⚠️ RISK PANEL
// ===============================
function renderRisk() {
    document.getElementById("risk-1h").innerText = mockRisk.last1h;
    document.getElementById("risk-3h").innerText = mockRisk.last3h;
    document.getElementById("risk-24h").innerText = mockRisk.last24h;
    document.getElementById("risk-7d").innerText = mockRisk.last7d;
}

// ===============================
// 📍 INCIDENTS MAP
// ===============================
function renderMapPoints() {
    incidentsLayer.clearLayers();

    for (let i = 0; i < 100; i++) {
        const lat = 37.75 + Math.random() * 0.1;
        const lon = -122.45 + Math.random() * 0.1;

        L.circleMarker([lat, lon], {
            radius: 5
        }).addTo(incidentsLayer);
    }
}

// ===============================
// 📱 MOBILE DRAG HANDLE
// ===============================
function initDragHandle() {
    const handle = document.getElementById("drag-handle");
    const sidebar = document.getElementById("sidebar");

    let isDragging = false;

    handle.addEventListener("touchstart", () => isDragging = true);
    handle.addEventListener("touchend", () => isDragging = false);

    handle.addEventListener("touchmove", (e) => {
        if (!isDragging) return;

        const touch = e.touches[0];
        const height = window.innerHeight - touch.clientY;

        sidebar.style.height = height + "px";
    });
}

// ===============================
// 🚀 INIT APP
// ===============================
function init() {
    updateKPIs();
    initTrendChart();
    initCategoryChart();
    renderDistricts();
    renderRisk();
    renderMapPoints();
    initDragHandle();
}

document.addEventListener("DOMContentLoaded", init);