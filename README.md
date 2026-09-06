# Project TTDN: Phân vùng Lớp phủ Đất (LULC) Huyện Giao Thủy

Hệ thống phân vùng ảnh vệ tinh viễn thám (Satellite Semantic Segmentation) đa kênh Sentinel-2 và ESA WorldCover v200 cho huyện Giao Thủy (Nam Định), chuẩn hóa 7 lớp nhãn sử dụng Deep Learning và MLOps.

---

## 📌 Tổng quan Giai đoạn 1 (Phase 1: PoC Pipeline)

Dự án đã hoàn thiện trọn vẹn 5 bước của Giai đoạn 1 theo tiêu chuẩn kỹ thuật nghiêm ngặt:

1. **Bước 1.1 (GEE Extractor):** Tải và căn chỉnh lưới tọa độ UTM EPSG:32648 cho cặp ảnh Sentinel-2 (4 kênh B2, B3, B4, B8) và ESA WorldCover v200 (độ phân giải 10m/pixel).
2. **Bước 1.2 (Class Remapping):** Ánh xạ từ nhãn ESA sang 7 lớp đối tượng địa lý bằng bảng tra cứu NumPy vectorized $O(n)$.
3. **Bước 1.3 (EDA & Class Weights):** Phân tích tương quan phổ, mở rộng AOI bao trùm Vườn Quốc gia Xuân Thủy để bảo toàn lớp Rừng ngập mặn (~6.08%), tính trọng số cân bằng tần suất trung vị (Median Frequency Balancing - MFB).
4. **Bước 1.4 (Spatial Block Split):** Phân chia tập dữ liệu theo 20 khối không gian địa lý (Lưới 4x5) với 14 khối Train (70%) và 6 khối Golden Test (30%), trích xuất 100 ảnh patch $256 \times 256$ chống rò rỉ dữ liệu (spatial leakage).
5. **Bước 1.5 (Data Version Control - DVC):** Tách biệt Code vs Data. Git quản lý mã nguồn, DVC quản lý toàn bộ dữ liệu ảnh thô và nhãn phân mảnh.

---

## 📂 Cấu trúc Thư mục

```text
project-TTDN/
├── .dvc/                        # Cấu hình Data Version Control
├── configs/
│   └── giao_thuy_expanded.geojson # Ranh giới địa lý mở rộng AOI
├── data/
│   ├── raw.dvc                  # Con trỏ DVC cho dữ liệu ảnh vệ tinh gốc
│   ├── processed.dvc            # Con trỏ DVC cho mask 7 lớp & 100 ảnh patches
│   └── interim/                 # Thư mục dữ liệu trung gian
├── process/                     # Báo cáo kỹ thuật chi tiết các bước
├── scripts/                     # Script kiểm tra, xuất ảnh preview
├── src/                         # Mã nguồn mô-đun chính
│   ├── class_mapping.py         # Bước 1.2: Remap 7 lớp
│   ├── config.py                # Cấu hình đường dẫn, thông số, 7 lớp
│   ├── data_split.py            # Bước 1.4: Spatial Block Split & Patching
│   ├── eda.py                   # Bước 1.3: EDA & Class Weights (MFB)
│   ├── gee_extractor.py         # Bước 1.1: Trích xuất GEE
│   └── gis_utils.py             # Tiện ích I/O GeoTIFF, Colormap
├── tests/                       # Unit tests (12/12 passing)
│   ├── test_class_mapping.py
│   ├── test_data_split.py
│   └── test_eda.py
├── .gitignore
├── requirements.txt
├── phase1_plan.md
└── roadmap.md
```

---

## 🛠️ Cài đặt Môi trường

```bash
# 1. Khởi tạo môi trường ảo
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# 2. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 3. Xác thực Google Earth Engine (nếu muốn tải ảnh mới từ GEE)
earthengine authenticate
```

---

## 📦 Quản lý Dữ liệu với DVC

Dữ liệu ảnh lớn (`data/raw/` và `data/processed/`) được quản lý bằng **DVC**.

```bash
# Kéo dữ liệu từ DVC remote về máy:
dvc pull

# Đẩy dữ liệu mới lên DVC remote sau khi cập nhật:
dvc add data/raw data/processed
dvc push
git add data/raw.dvc data/processed.dvc
git commit -m "chore: update data version"
```

---

## 🚀 Hướng dẫn Chạy Pipeline

### 1. Trích xuất ảnh Sentinel-2 & WorldCover từ GEE
```bash
python -m src.gee_extractor configs/giao_thuy_expanded.geojson data/raw/giao_thuy_expanded --start 2024-01-01 --end 2024-03-31 --project <YOUR_GCP_PROJECT_ID>
```

### 2. Ánh xạ 7 lớp nhãn (Remapping)
```bash
python -m src.class_mapping data/raw/giao_thuy_expanded_worldcover.tif data/processed/giao_thuy_expanded_mask7.tif
```

### 3. Phân tích thống kê EDA & Tính trọng số MFB Loss
```bash
python -m src.eda data/raw/giao_thuy_expanded_s2.tif data/processed/giao_thuy_expanded_mask7.tif data/processed/
```

### 4. Chia khối không gian (Spatial Split) & Cắt Patch
```bash
python -m src.data_split data/raw/giao_thuy_expanded_s2.tif data/processed/giao_thuy_expanded_mask7.tif data/processed/ --rows 4 --cols 5 --test-ratio 0.30 --patch-size 256 --stride 128
```

---

## 🧪 Chạy Kiểm thử (Unit Tests)

```bash
python -m unittest discover tests
```
Tất cả 12 bài unit test kiểm tra tính toàn vẹn của ánh xạ nhãn, tính trọng số MFB không chia cho 0, và thuật toán phân chia khối không gian chống rò rỉ dữ liệu.
