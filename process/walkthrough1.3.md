# Báo cáo Hoàn thành Bước 1.3: EDA & Tính Class Weights

Đã hoàn thành toàn diện **Bước 1.3** theo đúng tiêu chuẩn kỹ thuật trong [roadmap.md](file:///d:/project-TTDN/roadmap.md) và [phase1_plan.md](file:///d:/project-TTDN/phase1_plan.md).

---

## 1. Các thành phần đã triển khai

### 1.1. Module EDA & Median Frequency Balancing
- **[src/eda.py](file:///d:/project-TTDN/src/eda.py)**:
  - `compute_median_frequency_weights()`: Triển khai thuật toán **Median Frequency Balancing (MFB)** chỉ áp dụng cho các lớp foreground (1–6). Lớp 0 (Background) có trọng số cố định $= 0.0$ (phục vụ `ignore_index=0` của CrossEntropy / Combo Loss). Xử lý an toàn các lớp vắng mặt (như Rừng ngập mặn ở Giao Thủy có số pixel $= 0 \implies \text{weight} = 0.0$).
  - `generate_class_distribution_plot()`: Vẽ biểu đồ cột thể hiện diện tích (ha) và số pixel của 7 lớp với màu quy hoạch chuẩn (`CLASS_COLORS`).
  - `generate_spatial_overlay_plot()`: Tạo ảnh đối chiếu không gian 3 khung hình: Sentinel-2 True Color RGB, Mask 7 lớp và Spatial Overlay (độ mờ 50%).
  - `run_eda_pipeline()`: Điều phối toàn bộ quy trình, xuất cấu hình trọng số và tóm tắt thống kê.
  - Giao diện dòng lệnh CLI:
    ```bash
    python -m src.eda data/processed/giao_thuy_mask7.tif data/raw/giao_thuy_s2.tif --out-dir data/processed
    ```

### 1.2. Bộ kiểm thử đơn vị (Unit Tests)
- **[tests/test_eda.py](file:///d:/project-TTDN/tests/test_eda.py)**:
  - Kiểm tra tính toán trọng số MFB theo công thức lý thuyết trên dữ liệu tổng hợp.
  - Kiểm tra điều kiện biên: lớp 0 luôn bằng 0.0, lớp vắng mặt bằng 0.0, không xảy ra ZeroDivisionError.
  - Kiểm tra end-to-end quy trình sinh JSON và ảnh đồ thị trên raster giả lập.

---

## 2. Kết quả Tính toán Trọng số Lớp học (Giao Thủy)

Thuật toán Median Frequency Balancing tính toán trên $731,941$ pixel foreground (tần suất trung vị $\text{median\_freq} \approx 0.095918$ tại Lớp 5 - Cây lâu năm):

| Mã | Tên Lớp | Số Pixel | Tỷ Lệ (%) | Diện Tích (ha) | Trọng Số MFB ($w_c$) | Trọng Số Chuẩn Hóa | Ý Nghĩa Điều Tiết |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | Background | 749 | 0.10% | 7.49 | **0.0000** | 0.0000 | `ignore_index=0` (không tính loss) |
| **1** | Lúa | 422,333 | 57.64% | 4,223.33 | **0.1662** | 0.0432 | Giảm phạt lớp áp đảo (57.6%) |
| **2** | Khu dân cư | 200,290 | 27.34% | 2,002.90 | **0.3505** | 0.0912 | Giảm phạt lớp phổ biến (27.3%) |
| **3** | Thủy sản | 34,633 | 4.73% | 346.33 | **2.0271** | 0.5274 | Tăng trọng số nhẹ ($\times 2$) |
| **4** | Rừng ngập mặn | 0 | 0.00% | 0.00 | **0.0000** | 0.0000 | Vắng mặt tại AOI (trọng số 0) |
| **5** | Cây lâu năm & TV khác | 70,206 | 9.58% | 702.06 | **1.0000** | 0.2602 | **Lớp chuẩn trung vị (Median)** |
| **6** | Đồng muối / Đất trống | 4,479 | 0.61% | 44.79 | **15.6745** | 4.0780 | Tăng phạt mạnh lớp thiểu số ($\times 15.7$) |

Dữ liệu trọng số đã được lưu tại [data/processed/class_weights.json](file:///d:/project-TTDN/data/processed/class_weights.json), sẵn sàng nạp trực tiếp vào `torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device), ignore_index=0)`.

---

## 3. Biểu đồ Phân bố & Đối chiếu Không gian

![Phân bố diện tích 7 lớp sử dụng đất tại Giao Thủy](C:/Users/user/.gemini/antigravity-ide/brain/f05e09f0-c3a4-49c1-90d6-29cea7a75387/eda_class_distribution.png)

Các file ảnh kết xuất có độ phân giải cao tại [data/processed/](file:///d:/project-TTDN/data/processed/):
- **Biểu đồ phân bố**: [eda_class_distribution.png](file:///d:/project-TTDN/data/processed/eda_class_distribution.png)
- **Ảnh đối chiếu không gian (Sentinel-2 vs Mask 7 vs Overlay)**: [eda_spatial_overlay.png](file:///d:/project-TTDN/data/processed/eda_spatial_overlay.png)
- **Tóm tắt EDA định dạng JSON**: [eda_summary.json](file:///d:/project-TTDN/data/processed/eda_summary.json)

---

## 4. Kết quả Kiểm thử Toàn diện (All Tests Passed)

Chạy bộ unit test của toàn bộ dự án (`test_class_mapping` + `test_eda`):
```bash
.venv\Scripts\python.exe -m unittest discover tests -v
```
**Kết quả**:
```text
test_build_lookup_table_default (test_class_mapping.TestClassMapping.test_build_lookup_table_default) ... ok
test_build_lookup_table_validation (test_class_mapping.TestClassMapping.test_build_lookup_table_validation) ... ok
test_compute_class_histogram (test_class_mapping.TestClassMapping.test_compute_class_histogram) ... ok
test_remap_array_success (test_class_mapping.TestClassMapping.test_remap_array_success) ... ok
test_remap_raster_roundtrip (test_class_mapping.TestClassMapping.test_remap_raster_roundtrip) ... ok
test_compute_median_frequency_all_zero (test_eda.TestEDA.test_compute_median_frequency_all_zero) ... ok
test_compute_median_frequency_weights_standard (test_eda.TestEDA.test_compute_median_frequency_weights_standard) ... ok
test_run_eda_pipeline_integration (test_eda.TestEDA.test_run_eda_pipeline_integration) ... ok

Ran 8 tests in 2.863s - OK
```

---

## 5. Sẵn sàng cho Bước 1.4 (Spatial Split 70% Train / 30% Golden Test)

Bước 1.3 đã hoàn tất, cung cấp toàn bộ cơ sở về phân bố diện tích và trọng số cân bằng lớp. Tiếp theo là **Bước 1.4**:
- Chia dữ liệu theo khối không gian liền mạch (**Spatial Block Split**) $70\%$ Train / $30\%$ Golden Test trên cặp ảnh [giao_thuy_s2.tif](file:///d:/project-TTDN/data/raw/giao_thuy_s2.tif) và [giao_thuy_mask7.tif](file:///d:/project-TTDN/data/processed/giao_thuy_mask7.tif), tuyệt đối không tách ngẫu nhiên theo pixel nhằm loại bỏ spatial data leakage.
