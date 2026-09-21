# 🛰️ Hệ thống Giám sát Hiện trạng Sử dụng Đất qua Viễn thám & Deep Learning (WebGIS)

Hệ thống phân vùng và giám sát lớp phủ sử dụng đất (LULC) tự động từ ảnh vệ tinh đa phổ **Sentinel-2 L2A** và **ESA WorldCover 10m** cho huyện Giao Thủy (Nam Định), ứng dụng mô hình học sâu **ResNet-34 U-Net** và giao diện tương tác **WebGIS (FastAPI + Leaflet.js)**.

---

## 🌟 Điểm nổi bật của Dự án

- **Pipeline khép kín (End-to-End):** Từ thu nhận ảnh vệ tinh đa phổ, tiền xử lý, huấn luyện AI, hậu xử lý MMU đến ứng dụng WebGIS tương tác.
- **Chống rò rỉ dữ liệu (Zero Data Leakage):** Áp dụng kỹ thuật **Spatial Block Split (8x8 khối địa lý độc lập)** thay vì chia ngẫu nhiên.
- **Mô hình học sâu hiệu năng cao:** **ResNet-34 U-Net** kết hợp đầu vào 6 kênh phổ (B2, B3, B4, B8, NDVI, NDWI) và hàm mất mát kết hợp `Combo Loss (Weighted CE + Focal + Dice)`.
- **Kết quả thực nghiệm:** Đạt **53.65% Validation mIoU** và **53.76% Golden Test mIoU** (OA: **73.80%**, Kappa: **0.642**).
- **Hệ thống WebGIS hiện đại:** Giao diện Sleek Glassmorphism Dark Mode, tích hợp thanh trượt so sánh vệ tinh (Swipe Tool), tra cứu pixel tức thì (< 30ms) và xuất báo cáo diện tích CSV.
- **Độ tin cậy kỹ thuật:** Đạt chuẩn **74/74 Unit Tests (100% Passed)**.

---

## 🗺️ Bảng 6 Lớp Đất & Kết quả Đo đạc Huyện Giao Thủy

| Mã | Loại Đất Hiện Trạng | Màu Nhận Diện | Diện Tích (ha) | Tỷ Lệ (%) | IoU Đạt Được |
| :-: | :--- | :-: | :-: | :-: | :-: |
| **1** | **Lúa nước** | Vàng (`#ffd700`) | 8,011.95 | 35.51% | **72.6%** |
| **2** | **Khu dân cư** | Đỏ đô (`#dc143c`) | 3,962.20 | 17.56% | **51.2%** |
| **3** | **Thủy sản (Đầm tôm/ngao)** | Xanh biển (`#00bfff`) | 6,394.14 | 28.34% | **77.8%** |
| **4** | **Rừng ngập mặn (Xuân Thủy)** | Xanh lá đậm (`#006400`) | 1,378.42 | 6.11% | **56.4%** |
| **5** | **Cây lâu năm & Vườn nhà** | Xanh rừng (`#228b22`) | 2,416.27 | 10.71% | **17.3%** |
| **6** | **Đồng muối / Đất trống** | Cam đất sáng (`#f97316`)| 402.39 | 1.78% | **10.5%** |
| **Σ** | **Toàn huyện Giao Thủy** | — | **22,565.37 ha** | **100.0%** | **mIoU: 53.65%** |

---

## 📂 Cấu trúc Thư mục Dự án

```text
project-TTDN/
├── configs/                # Cấu hình huấn luyện (YAML) & Bounding Box (GeoJSON)
├── data/
│   ├── raw/                # Ảnh gốc Sentinel-2 và nhãn ESA WorldCover (.tif)
│   └── processed/          # Mask 7 lớp & Metadata khối không gian
├── outputs/
│   ├── logs/               # Biểu đồ học tập, Confusion Matrix, Per-class IoU
│   └── maps/               # Bản đồ dự đoán LULC (GeoTIFF chuẩn WGS84, PNG, CSV)
├── src/
│   ├── api/                # FastAPI Backend RESTful (6 endpoints) & Web Overlays
│   ├── model.py            # Kiến trúc ResNet-34 U-Net 6 kênh đầu vào
│   ├── trainer.py          # Huấn luyện mô hình, Mixed Precision FP16, Early Stopping
│   ├── evaluator.py        # Tính toán mIoU, OA, Kappa, Confusion Matrix
│   ├── predict.py          # Suy luận trượt toàn cảnh & Lọc nhiễu MMU 500m²
│   ├── losses.py           # Combo Loss (Weighted CE + Focal Loss + Dice Loss)
│   └── transforms.py       # Data Augmentation hình học và quang học
├── tests/                  # 9 bộ kiểm thử tự động toàn diện (74 tests)
├── web/                    # Giao diện WebGIS Dashboard (HTML, CSS, JS, Assets)
├── requirements.txt        # Danh sách thư viện phụ thuộc
└── README.md
```

---

## 🚀 Cài đặt & Khởi chạy Nhanh

### 1. Cài đặt Môi trường
```powershell
# Tạo và kích hoạt môi trường ảo
python -m venv .venv
.venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Khởi chạy Ứng dụng WebGIS (Demo)
Chạy lệnh duy nhất để khởi động máy chủ:
```powershell
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```
Mở trình duyệt web truy cập: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

- **Bản đồ:** Chuyển đổi lớp nền Esri Satellite / OSM và xem lớp phủ AI 6 màu.
- **Thanh trượt Swipe:** So sánh trực quan giữa ảnh vệ tinh thật và bản đồ AI.
- **Click tra cứu:** Bấm bất kỳ điểm nào trên bản đồ để xem tọa độ, loại đất và mô tả quy hoạch.
- **Thống kê:** Chọn xã từ menu dropdown để xem biểu đồ tròn tự động cập nhật và tải file CSV.

### 3. Huấn luyện lại Mô hình (Tùy chọn)
```powershell
python -m src.trainer --config configs/train_config.yaml
```

### 4. Chạy Kiểm thử Tự động (Tests)
```powershell
python -m unittest discover -s tests -p "test_*.py"
```
*(Kết quả: 74/74 tests pass 100%)*
