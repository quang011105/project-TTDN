/**
 * LULC SATELLITE MONITORING — WEBGIS APP LOGIC
 * Framework: Vanilla JS, Leaflet.js, Chart.js
 */

document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------------------
    // 1. STATE & GLOBAL VARIABLES
    // -------------------------------------------------------------------------
    let map = null;
    let baseLayers = {};
    let currentBaseLayer = null;
    let lulcOverlay = null;
    let s2RgbOverlay = null;
    let s2CirOverlay = null;
    let sideBySideControl = null;
    let isSwipeActive = false;
    let chartInstance = null;
    let currentBounds = null;
    let currentUnitId = "giao_thuy";
    let inspectMarker = null;

    const CLASS_PALETTE = {
        1: { name: "Lúa", color: "#ffd700" },
        2: { name: "Khu dân cư", color: "#dc143c" },
        3: { name: "Thủy sản", color: "#00bfff" },
        4: { name: "Rừng ngập mặn", color: "#006400" },
        5: { name: "Cây lâu năm", color: "#228b22" },
        6: { name: "Đồng muối / Trống", color: "#f97316" }
    };

    // -------------------------------------------------------------------------
    // 2. INITIALIZE LEAFLET MAP & LAYERS
    // -------------------------------------------------------------------------
    async function initMap() {
        // Tọa độ tâm mặc định huyện Giao Thủy
        map = L.map('map', {
            center: [20.2502, 106.4679],
            zoom: 12,
            minZoom: 10,
            maxZoom: 18,
            zoomControl: false
        });

        L.control.zoom({ position: 'topright' }).addTo(map);

        // Base Tile Layers
        baseLayers = {
            esri_satellite: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS',
                maxZoom: 18
            }),
            osm: L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; OpenStreetMap contributors',
                maxZoom: 18
            }),
            carto_dark: L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; CARTO &copy; OpenStreetMap',
                maxZoom: 18
            })
        };

        currentBaseLayer = baseLayers.esri_satellite;
        currentBaseLayer.addTo(map);

        // Nạp Map Metadata từ Backend API
        try {
            const resp = await fetch('/api/map-metadata');
            const result = await resp.json();
            if (result.status === 'success') {
                const meta = result.data;
                currentBounds = meta.leaflet_bounds; // [[lat_min, lon_min], [lat_max, lon_max]]

                // Tạo các Overlays (kèm query timestamp để tránh browser cache ảnh cũ)
                const cacheBuster = `?t=${Date.now()}`;
                s2RgbOverlay = L.imageOverlay(meta.layers.s2_rgb + cacheBuster, currentBounds, {
                    opacity: 1.0,
                    interactive: true
                });

                s2CirOverlay = L.imageOverlay(meta.layers.s2_cir + cacheBuster, currentBounds, {
                    opacity: 1.0,
                    interactive: true
                });

                lulcOverlay = L.imageOverlay(meta.layers.lulc + cacheBuster, currentBounds, {
                    opacity: 0.85,
                    interactive: true
                });

                // Mặc định thêm lớp LULC AI lên bản đồ
                lulcOverlay.addTo(map);
            }
        } catch (err) {
            console.error('Lỗi khi nạp Map Metadata:', err);
        }

        // Sự kiện di chuột cập nhật tọa độ
        map.on('mousemove', (e) => {
            const lat = e.latlng.lat.toFixed(4);
            const lng = e.latlng.lng.toFixed(4);
            document.getElementById('coordsChip').innerHTML = `<i class="fa-solid fa-crosshairs"></i> Tọa độ: ${lat}°N, ${lng}°E`;
        });

        // Sự kiện click chuột tra cứu pixel
        map.on('click', handleMapClick);

        renderLegend();
    }

    // -------------------------------------------------------------------------
    // 3. LAYER CONTROLS & OPACITY
    // -------------------------------------------------------------------------
    function setupLayerControls() {
        const lulcToggle = document.getElementById('layerLulcToggle');
        const s2RgbToggle = document.getElementById('layerS2RgbToggle');
        const s2CirToggle = document.getElementById('layerS2CirToggle');
        const opacitySlider = document.getElementById('lulcOpacity');
        const opacityVal = document.getElementById('opacityVal');
        const baseMapSelect = document.getElementById('baseMapSelect');

        lulcToggle.addEventListener('change', (e) => {
            if (e.target.checked) {
                if (lulcOverlay) map.addLayer(lulcOverlay);
            } else {
                if (lulcOverlay) map.removeLayer(lulcOverlay);
            }
        });

        s2RgbToggle.addEventListener('change', (e) => {
            if (e.target.checked) {
                if (s2RgbOverlay) map.addLayer(s2RgbOverlay);
            } else {
                if (s2RgbOverlay) map.removeLayer(s2RgbOverlay);
            }
        });

        s2CirToggle.addEventListener('change', (e) => {
            if (e.target.checked) {
                if (s2CirOverlay) map.addLayer(s2CirOverlay);
            } else {
                if (s2CirOverlay) map.removeLayer(s2CirOverlay);
            }
        });

        opacitySlider.addEventListener('input', (e) => {
            const val = e.target.value;
            opacityVal.textContent = `${val}%`;
            if (lulcOverlay) {
                lulcOverlay.setOpacity(val / 100.0);
            }
        });

        baseMapSelect.addEventListener('change', (e) => {
            const selected = e.target.value;
            if (currentBaseLayer) map.removeLayer(currentBaseLayer);
            currentBaseLayer = baseLayers[selected];
            currentBaseLayer.addTo(map);
            if (currentBaseLayer.bringToBack) currentBaseLayer.bringToBack();
        });
    }

    // -------------------------------------------------------------------------
    // 4. SWIPE / SIDE-BY-SIDE COMPARISON TOOL
    // -------------------------------------------------------------------------
    function setupSwipeControl() {
        const swipeBtn = document.getElementById('swipeToggleBtn');

        swipeBtn.addEventListener('click', () => {
            if (!s2RgbOverlay || !lulcOverlay) return;

            if (!isSwipeActive) {
                // Bật Swipe: Left là S2 RGB, Right là AI LULC
                if (!map.hasLayer(s2RgbOverlay)) map.addLayer(s2RgbOverlay);
                if (!map.hasLayer(lulcOverlay)) map.addLayer(lulcOverlay);

                document.getElementById('layerLulcToggle').checked = true;
                document.getElementById('layerS2RgbToggle').checked = true;

                if (L.control.sideBySide) {
                    sideBySideControl = L.control.sideBySide(s2RgbOverlay, lulcOverlay);
                    sideBySideControl.addTo(map);
                }

                swipeBtn.innerHTML = `<i class="fa-solid fa-xmark"></i> Tắt Thanh trượt So sánh`;
                swipeBtn.classList.add('active');
                isSwipeActive = true;
            } else {
                // Tắt Swipe
                if (sideBySideControl) {
                    map.removeControl(sideBySideControl);
                    sideBySideControl = null;
                }
                if (map.hasLayer(s2RgbOverlay)) map.removeLayer(s2RgbOverlay);
                document.getElementById('layerS2RgbToggle').checked = false;

                swipeBtn.innerHTML = `<i class="fa-solid fa-arrows-left-right"></i> Bật Thanh trượt So sánh`;
                swipeBtn.classList.remove('active');
                isSwipeActive = false;
            }
        });
    }

    // -------------------------------------------------------------------------
    // 5. POINT INSPECTION (CLICK MAP TO QUERY PIXEL)
    // -------------------------------------------------------------------------
    async function handleMapClick(e) {
        const lat = e.latlng.lat;
        const lon = e.latlng.lng;
        const popup = document.getElementById('inspectPopup');

        document.getElementById('mapStatusChip').innerHTML = `<i class="fa-solid fa-spinner fa-spin text-info"></i> Đang truy vấn pixel...`;

        try {
            const resp = await fetch(`/api/inspect?lat=${lat}&lon=${lon}`);
            const res = await resp.json();
            
            if (res.status === 'success') {
                const data = res.data;
                popup.classList.remove('hidden');

                // Hiển thị marker vị trí click trên bản đồ
                if (inspectMarker) {
                    map.removeLayer(inspectMarker);
                }
                inspectMarker = L.circleMarker([lat, lon], {
                    radius: 7,
                    color: '#ffffff',
                    weight: 2.5,
                    fillColor: data.color || '#f97316',
                    fillOpacity: 1.0
                }).addTo(map);

                const badge = document.getElementById('inspectClassBadge');
                badge.textContent = data.class_name;
                badge.style.backgroundColor = data.color;
                badge.style.color = '#ffffff';

                document.getElementById('inspectCoords').textContent = `${data.lat}, ${data.lon}`;
                document.getElementById('inspectPixelIdx').textContent = data.inside ? `Hàng ${data.row}, Cột ${data.col}` : 'Ngoài vùng ảnh';
                document.getElementById('inspectDesc').textContent = data.description;

                document.getElementById('mapStatusChip').innerHTML = `<i class="fa-solid fa-circle-check text-success"></i> Đã chọn: ${data.class_name}`;
            }
        } catch (err) {
            console.error('Lỗi tra cứu pixel:', err);
            document.getElementById('mapStatusChip').innerHTML = `<i class="fa-solid fa-circle-exclamation text-danger"></i> Lỗi tra cứu`;
        }
    }

    document.getElementById('closeInspectBtn').addEventListener('click', () => {
        document.getElementById('inspectPopup').classList.add('hidden');
        if (inspectMarker) {
            map.removeLayer(inspectMarker);
            inspectMarker = null;
        }
    });

    // -------------------------------------------------------------------------
    // 6. LOAD ADMINISTRATIVE UNITS & STATS
    // -------------------------------------------------------------------------
    async function loadAdminUnits() {
        const select = document.getElementById('adminSelect');
        const descBox = document.getElementById('unitDescBox');

        try {
            const resp = await fetch('/api/administrative-units');
            const res = await resp.json();

            if (res.status === 'success') {
                select.innerHTML = '';
                res.data.forEach(unit => {
                    const opt = document.createElement('option');
                    opt.value = unit.id;
                    opt.textContent = unit.name;
                    select.appendChild(opt);
                });

                select.value = 'giao_thuy';
                descBox.textContent = res.data[0].description;

                // Nạp stats cho toàn huyện ban đầu
                fetchAndRenderStats('giao_thuy');
            }
        } catch (err) {
            console.error('Lỗi nạp đơn vị hành chính:', err);
        }

        select.addEventListener('change', async (e) => {
            currentUnitId = e.target.value;
            try {
                const resp = await fetch(`/api/administrative-units/${currentUnitId}`);
                const res = await resp.json();
                if (res.status === 'success') {
                    const unit = res.data;
                    descBox.textContent = unit.description;

                    // Di chuyển bản đồ mượt mà đến xã được chọn
                    if (unit.bbox && map) {
                        map.flyToBounds(unit.bbox, {
                            duration: 1.2,
                            padding: [40, 40]
                        });
                    }

                    // Nạp số liệu thống kê mới
                    fetchAndRenderStats(currentUnitId);
                }
            } catch (err) {
                console.error(err);
            }
        });
    }

    // -------------------------------------------------------------------------
    // 7. FETCH STATS & RENDER CHARTS
    // -------------------------------------------------------------------------
    async function fetchAndRenderStats(unitId) {
        const container = document.getElementById('classListContainer');
        container.innerHTML = `<div class="loading-spinner"><i class="fa-solid fa-spinner fa-spin"></i> Đang tính toán diện tích...</div>`;

        try {
            const resp = await fetch(`/api/lulc-stats?unit_id=${unitId}`);
            const res = await resp.json();

            if (res.status === 'success') {
                const stats = res.data;
                
                // Cập nhật thẻ tổng
                document.getElementById('totalAreaHa').textContent = stats.total_area_ha.toLocaleString('vi-VN');
                document.getElementById('totalAreaKm2').textContent = stats.total_area_km2.toLocaleString('vi-VN');

                // Render danh sách thẻ KPI
                container.innerHTML = '';
                const chartLabels = [];
                const chartData = [];
                const chartColors = [];

                stats.classes.forEach(c => {
                    chartLabels.push(c.name);
                    chartData.push(c.percentage);
                    chartColors.push(c.color);

                    const card = document.createElement('div');
                    card.className = 'class-kpi-card';
                    card.innerHTML = `
                        <div class="kpi-row-top">
                            <div class="kpi-title-flex">
                                <span class="kpi-dot" style="background-color: ${c.color}"></span>
                                <span class="kpi-name">${c.name}</span>
                            </div>
                            <span class="kpi-pct">${c.percentage}%</span>
                        </div>
                        <div class="kpi-row-bottom">
                            <span>Diện tích: <strong>${c.area_ha.toLocaleString('vi-VN')} ha</strong></span>
                            <span>(~${c.area_km2.toLocaleString('vi-VN')} km²)</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${c.percentage}%; background-color: ${c.color}"></div>
                        </div>
                    `;
                    container.appendChild(card);
                });

                // Cập nhật biểu đồ Doughnut
                renderDoughnutChart(chartLabels, chartData, chartColors);
            }
        } catch (err) {
            console.error('Lỗi tính diện tích:', err);
            container.innerHTML = `<div class="text-danger">Không thể nạp dữ liệu thống kê.</div>`;
        }
    }

    function renderDoughnutChart(labels, data, colors) {
        const ctx = document.getElementById('lulcChart').getContext('2d');

        if (chartInstance) {
            chartInstance.destroy();
        }

        chartInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: colors,
                    borderWidth: 1,
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    hoverOffset: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return ` ${context.label}: ${context.raw}%`;
                            }
                        }
                    }
                },
                cutout: '68%'
            }
        });
    }

    // -------------------------------------------------------------------------
    // 8. RENDER LEGEND
    // -------------------------------------------------------------------------
    function renderLegend() {
        const body = document.getElementById('legendBody');
        body.innerHTML = '';
        for (const [cid, info] of Object.entries(CLASS_PALETTE)) {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <span class="legend-color-box" style="background-color: ${info.color}"></span>
                <span>${info.name}</span>
            `;
            body.appendChild(item);
        }

        const collapseBtn = document.getElementById('legendCollapseBtn');
        collapseBtn.addEventListener('click', () => {
            body.style.display = body.style.display === 'none' ? 'block' : 'none';
            collapseBtn.innerHTML = body.style.display === 'none' ? 
                '<i class="fa-solid fa-chevron-up"></i>' : '<i class="fa-solid fa-chevron-down"></i>';
        });
    }

    // -------------------------------------------------------------------------
    // 9. EXPORT CSV BUTTON & THEME TOGGLE
    // -------------------------------------------------------------------------
    document.getElementById('exportCsvBtn').addEventListener('click', () => {
        window.location.href = `/api/export-csv?unit_id=${currentUnitId}`;
    });

    const themeToggleBtn = document.getElementById('themeToggleBtn');
    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        themeToggleBtn.innerHTML = newTheme === 'dark' ? 
            '<i class="fa-solid fa-sun"></i>' : '<i class="fa-solid fa-moon"></i>';
    });

    // -------------------------------------------------------------------------
    // RUN APP
    // -------------------------------------------------------------------------
    initMap();
    setupLayerControls();
    setupSwipeControl();
    loadAdminUnits();
});
