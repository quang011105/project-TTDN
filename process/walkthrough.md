# Báo cáo Hoàn thành Bước 1.2: Ánh xạ Nhãn (Class Mapping)

Đã hoàn thành toàn diện **Bước 1.2** theo đúng tiêu chuẩn thiết kế trong [roadmap.md](file:///d:/project-TTDN/roadmap.md) và [phase1_plan.md](file:///d:/project-TTDN/phase1_plan.md).

---

## 1. Các thành phần đã triển khai

### Mã nguồn & Cấu hình
1. **[src/config.py](file:///d:/project-TTDN/src/config.py)**:
   - Cập nhật từ điển ánh xạ `ESA_TO_LOCAL_CLASS` chuẩn hóa các mã:
     - $40 \to 1$ (Lúa)
     - $50 \to 2$ (Khu dân cư)
     - $80 \to 3$ (Thủy sản)
     - $95 \to 4$ (Rừng ngập mặn)
     - $10, 20, 30 \to 5$ (Cây lâu năm & Thực vật khác)
     - $60 \to 6$ (Đồng muối / Đất trống)
     - Các mã còn lại ($0, 70, 90, 100\dots$) mặc định về $0$ (Background).
   - Thêm từ điển `CLASS_NAMES` và bảng mã màu `CLASS_COLORS` phục vụ trực quan hóa và nhúng Colormap chuẩn GIS.
2. **[src/class_mapping.py](file:///d:/project-TTDN/src/class_mapping.py)**:
   - `build_lookup_table()`: Xây dựng mảng tra cứu NumPy 256 phần tử kiểu `uint8`.
   - `remap_array()`: Ánh xạ vectorized $O(1)$ mỗi pixel (không dùng vòng lặp Python, tránh lãng phí RAM).
   - `compute_class_histogram()` & `format_histogram_report()`: Tính toán và xuất báo cáo thống kê diện tích/tần suất.
   - `remap_worldcover_raster()`: Đọc GeoTIFF gốc, remap, nhúng colormap, kiểm tra ràng buộc không gian và lưu GeoTIFF `uint8` tại `data/processed/`.
   - `save_colored_preview()`: Xuất file ảnh PNG trực quan hóa mask với bảng màu chuẩn.
   - CLI tiện ích: `python -m src.class_mapping <input_tif> <output_tif> [--preview <preview_png>]`.
3. **[tests/test_class_mapping.py](file:///d:/project-TTDN/tests/test_class_mapping.py)**:
   - 5 unit tests kiểm thử: tạo LUT, kiểm tra biên [0, 255], remap ma trận, tính histogram và roundtrip đọc-ghi GeoTIFF thực tế qua `rasterio`.

---

## 2. Kết quả Kiểm thử & Xác thực Thực tế

### 2.1. Kiểm thử đơn vị (Unit Tests)
Chạy lệnh:
```bash
.venv\Scripts\python.exe -m unittest tests/test_class_mapping.py -v
```
**Kết quả**:
```text
test_build_lookup_table_default (tests.test_class_mapping.TestClassMapping.test_build_lookup_table_default) ... ok
test_build_lookup_table_validation (tests.test_class_mapping.TestClassMapping.test_build_lookup_table_validation) ... ok
test_compute_class_histogram (tests.test_class_mapping.TestClassMapping.test_compute_class_histogram) ... ok
test_remap_array_success (tests.test_class_mapping.TestClassMapping.test_remap_array_success) ... ok
test_remap_raster_roundtrip (tests.test_class_mapping.TestClassMapping.test_remap_raster_roundtrip) ... ok

Ran 5 tests in 0.386s - OK
```

### 2.2. Kiểm thử dữ liệu thực tế (xã Giao Thủy)
Chạy pipeline remap trên raster thực tế:
```bash
python -m src.class_mapping data/raw/giao_thuy_worldcover.tif data/processed/giao_thuy_mask7.tif --preview data/processed/preview_giao_thuy_mask7.png
```

- **Kiểm tra khớp lưới (`assert_rasters_aligned`)**:
  - `giao_thuy_s2.tif` $\leftrightarrow$ `giao_thuy_mask7.tif`: **PASSED** (Khớp tuyệt đối CRS `EPSG:32648`, Transform `(10.0, 0.0, 643120.0, 0.0, -10.0, 2245420.0)`, Shape `(630, 1163)`).
- **Kiểm tra kiểu dữ liệu & tập giá trị lớp**:
  - Kiểu dữ liệu: `uint8`.
  - Tập giá trị pixel: $\{0, 1, 2, 3, 5, 6\} \subseteq \{0, 1, 2, 3, 4, 5, 6\}$ (Không có giá trị bất thường).

### 2.3. Bảng thống kê diện tích 7 lớp (Giao Thủy)

| Mã | Tên Lớp | Số Pixel | Tỷ Lệ (%) | Diện Tích (ha) |
| :---: | :--- | :---: | :---: | :---: |
| **0** | Background | 749 | 0.10% | 7.49 |
| **1** | Lúa | 422,333 | 57.64% | 4,223.33 |
| **2** | Khu dân cư | 200,290 | 27.34% | 2,002.90 |
| **3** | Thủy sản | 34,633 | 4.73% | 346.33 |
| **4** | Rừng ngập mặn | 0 | 0.00% | 0.00 |
| **5** | Cây lâu năm & Thực vật khác | 70,206 | 9.58% | 702.06 |
| **6** | Đồng muối / Đất trống | 4,479 | 0.61% | 44.79 |
| **Tổng** | | **732,690** | **100.00%** | **7,326.90** |

---

## 3. Sẵn sàng cho Bước 1.3 (EDA & Mất cân bằng lớp)

Các dữ liệu đầu vào chuẩn đã sẵn sàng:
- Ảnh quang học composite: [data/raw/giao_thuy_s2.tif](file:///d:/project-TTDN/data/raw/giao_thuy_s2.tif)
- Mask 7 lớp chuẩn: [data/processed/giao_thuy_mask7.tif](file:///d:/project-TTDN/data/processed/giao_thuy_mask7.tif)
- Ảnh màu preview kiểm tra: [data/processed/preview_giao_thuy_mask7.png](file:///d:/project-TTDN/data/processed/preview_giao_thuy_mask7.png)
