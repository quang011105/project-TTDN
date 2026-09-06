# 📋 Prompt: Kế hoạch Triển khai Giai đoạn 1 - PoC & Chuẩn bị Dữ liệu Không gian

Tài liệu thiết kế kỹ thuật (Technical Design Document) cho **Giai đoạn 1**, khớp lộ trình 12 tuần / 1 kỹ sư. Ground Truth = **ESA WorldCover v200**. Không SAM, không vẽ ranh giới thủ công, không gán nhãn chuyên gia.

---

**Ngữ cảnh dự án:** "Hệ thống Giám sát Quy hoạch Sử dụng đất qua Viễn thám". Vai trò: Data Engineer & Geospatial ML Expert. Phạm vi: **Giai đoạn 1: PoC & Chuẩn bị Dữ liệu Không gian (Cấp Xã)** — tuần 1–2 của toàn hệ thống.

**Mục tiêu cốt lõi:** ETL đồng bộ Sentinel-2 và WorldCover; ánh xạ 11 lớp ESA → 7 lớp (0–6); Golden Test không spatial leakage; versioning bằng DVC.

## 1. Cấu trúc Thư mục (Project Organization)

```text
project-TTDN/
├── data/
│   ├── raw/             # GeoTIFF Sentinel-2 + ESA WorldCover (mã gốc 10–100)
│   ├── interim/         # Raster đã clip/align, chưa remap
│   └── processed/       # Mask 7 lớp, train/test spatial split
├── notebooks/
│   ├── 01_gee_download.ipynb      # Sentinel-2 + WorldCover
│   ├── 02_class_mapping.ipynb     # ESA → 7 lớp NumPy
│   └── 03_eda_and_split.ipynb
├── src/
│   ├── config.py           # ROI xã, khoảng thời gian, CRS, bảng ánh xạ
│   ├── gee_extractor.py    # GEE: S2 composite + WorldCover v200
│   ├── class_mapping.py    # Vectorized remap 11 → 7
│   ├── gis_utils.py        # Align, resample, windowed I/O
│   └── data_split.py       # Spatial block split
├── .dvc/
├── .gitignore
├── requirements.txt
└── README.md
```

## 2. Chi tiết Các bước Triển khai & Yêu cầu Kỹ thuật

### Bước 1.1: Trích xuất ảnh Sentinel-2 và nhãn ESA WorldCover (GEE)

- **Nhiệm vụ:** Một pipeline `earthengine-api` tải **cùng lúc, cùng ROI** ảnh quang học và bản đồ nhãn toàn cầu.
- **Yêu cầu kỹ thuật:**
  - Clip theo GeoJSON ranh giới xã.
  - Sentinel-2 L2A: 3–6 tháng mùa khô; mask mây QA60 hoặc SCL; `median()` → composite sạch mây.
  - ESA WorldCover **v200** (10 m): clip cùng geometry; **resample/reproject** khớp grid Sentinel-2 (nearest-neighbor cho nhãn rời rạc).
  - Xuất hai GeoTIFF: `*_s2.tif`, `*_worldcover.tif` (CRS: EPSG:32648 UTM 48N hoặc EPSG:4326 — thống nhất một CRS).
  - Không tải SAM, không polygonize thửa đất.

### Bước 1.2: Ánh xạ nhãn (Class Mapping)

- **Nhiệm vụ:** Ép 11 mã ESA về ma trận nguyên 7 kênh (giá trị pixel 0–6) bằng NumPy vectorized (`np.vectorize` tránh vòng lặp Python; ưu tiên lookup table `np.zeros(256)` + index).
- **Bảng ánh xạ bắt buộc:**

| Mã ESA WorldCover | Lớp đích | Tên lớp |
| :--- | :---: | :--- |
| 40 | 1 | Lúa |
| 50 | 2 | Khu dân cư |
| 80 | 3 | Thủy sản |
| 95 | 4 | Rừng ngập mặn |
| 10, 20, 30 | 5 | Cây lâu năm & Thực vật khác |
| 60 | 6 | Đồng muối / Đất trống |
| 70, 90, 100 (và mọi mã khác) | 0 | Background |

- **Yêu cầu kỹ thuật:**
  - Input: raster WorldCover đã align. Output: GeoTIFF mask `uint8`, nodata = 0 hoặc flag riêng — nhưng **giá trị học** của nền là 0.
  - Kiểm tra: `np.unique` chỉ thuộc `{0,1,2,3,4,5,6}`; ghi log histogram trước/sau remap.
  - Module: `src/class_mapping.py`.

### Bước 1.3: EDA & Mất cân bằng

- **Nhiệm vụ:** Đặc tính phân bố 7 lớp sau remap trên cấp xã.
- **Yêu cầu kỹ thuật:**
  - Biểu đồ diện tích / số pixel từng lớp 0–6.
  - Heatmap (hoặc RGB overlay) trên bản đồ nền.
  - Class weights **chỉ lớp 1–6** (Median Frequency Balancing). Lớp 0 không đưa vào trọng số — Giai đoạn 2 dùng `ignore_index=0`.

### Bước 1.4: Tách dữ liệu theo không gian (Spatial Split)

- **Nhiệm vụ:** Train 70% / Golden Test 30% trên **cùng** cặp ảnh–mask 7 lớp.
- **Yêu cầu kỹ thuật:**
  - Không dùng `train_test_split` ngẫu nhiên theo pixel.
  - Spatial Block Split: 30% là một (hoặc vài) khối địa lý liền mạch, tách khỏi train.
  - Cắt patch train/test **sau** khi xác định khối, giữ geotransform.

### Bước 1.5: Quản lý phiên bản dữ liệu (DVC)

- **Nhiệm vụ:** Tái lập cặp S2 + WorldCover + mask 7 lớp.
- **Yêu cầu kỹ thuật:**
  - `dvc init`; remote local / Drive / S3.
  - Track `data/raw` và `data/processed`.
  - Commit file `.dvc` lên Git.

---

## 3. Lựa chọn Công nghệ & Lý do (Technology Stack & Justification)

| Thành phần | Công nghệ chọn | Lý do |
| :--- | :--- | :--- |
| **Nguồn ảnh** | `Google Earth Engine` | Composite + mask mây trên server Google; không tải SAFE/Copernicus về máy. |
| **Ground Truth** | `ESA WorldCover v200` | Nhãn 10 m toàn cầu, không gán tay, không SAM; đủ cho PoC 12 tuần. |
| **Ánh xạ lớp** | `NumPy` lookup table | Remap raster lớn O(n), không vòng lặp pixel Python. |
| **GIS Python** | `GeoPandas`, `Rasterio`, `Shapely` | Windowed I/O GeoTIFF; align lưới ảnh–nhãn. |
| **Data Versioning** | `DVC` | Ảnh/nhãn hàng trăm MB–GB không thuộc Git blob. |

QGIS chỉ dùng để **kiểm tra trực quan** overlay S2 / WorldCover / mask 7 lớp — không phải công cụ gán nhãn.

---

## 4. Tiêu chuẩn Thẩm mỹ & Hiệu năng Code

1. **Thẩm mỹ:** PEP 8 (`black`, `flake8`); Type Hinting mọi hàm; Google Style docstrings.
2. **Hiệu năng:** Windowed reading `rasterio`; remap vectorized; không `iterrows()` trên GeoDataFrame.

---

**Hợp đồng với Giai đoạn 2:** Mask processed có đúng 7 giá trị 0–6. Huấn luyện chỉ trên xã. Loss Cross-Entropy **bắt buộc** `ignore_index=0`.

**Yêu cầu hành động đối với AI:** Khi bắt đầu code, viết **Bước 1.1 (`gee_extractor.py`)** trước: cấu trúc thư mục, Type Hinting, exception handling, tải **cặp** Sentinel-2 + WorldCover v200. Đợi review rồi mới **Bước 1.2 (class mapping)**.
