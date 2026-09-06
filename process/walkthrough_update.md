# Báo cáo Hoàn thành Bước 1.3: Cập nhật Dữ liệu Mở rộng & Phân tích 7 Lớp

Đã cập nhật thành công toàn bộ chuỗi pipeline từ **Bước 1.1 $\to$ Bước 1.2 $\to$ Bước 1.3** trên vùng nghiên cứu mở rộng (bao trùm cả vùng nội đồng Giao Thủy và dải Rừng ngập mặn Vườn quốc gia Xuân Thủy).

---

## 1. Kết quả Cập nhật Vùng Nghiên cứu Mở rộng (AOI)

- **File GeoJSON ranh giới**: [data/raw/giao_thuy_expanded.geojson](file:///d:/project-TTDN/data/raw/giao_thuy_expanded.geojson)
  - Tọa độ Bounding Box: Kinh độ $106.3707^\circ \to 106.5650^\circ E$, Vĩ độ $20.2000^\circ \to 20.3003^\circ N$.
  - Mở rộng về phía Đông Nam ven biển để bao trọn Cồn Lu, Cồn Ngạn và dải rừng ngập mặn cửa sông Ba Lạt.
- **Kích thước ảnh & Kiểm tra khớp lưới**:
  - `giao_thuy_expanded_s2.tif`: Kích thước $(1130, 2040)$, 4 kênh phổ (B2, B3, B4, B8), kiểu `uint16`.
  - `giao_thuy_expanded_worldcover.tif`: Kích thước $(1130, 2040)$, 1 kênh, kiểu `uint8`.
  - Kiểm tra căn chỉnh `assert_rasters_aligned`: **PASSED** (Khớp tuyệt đối CRS `EPSG:32648` và Transform 10m).

---

## 2. Bảng Thống kê Diện tích 7 Lớp Mới (Có Đầy Đủ Rừng Ngập Mặn)

Tổng diện tích phân tích: **23,052 ha** ($2,305,200$ pixel):

| Mã Lớp | Tên Lớp Sử Dụng Đất | Số Pixel | Tỷ Lệ (%) | Diện Tích (ha) | Trọng Số MFB ($w_c$) | Ý Nghĩa Điều Tiết Loss |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **0** | Background | 3,737 | 0.16% | 37.37 | **0.0000** | `ignore_index=0` (loại khỏi hàm mất mát) |
| **1** | Lúa | 923,822 | 40.08% | 9,238.22 | **0.3307** | Giảm phạt nhẹ lớp đất lúa đa số |
| **2** | Khu dân cư | 396,145 | 17.18% | 3,961.45 | **0.7711** | Cân bằng vùng dân cư / đô thị |
| **3** | Thủy sản | 586,186 | 25.43% | 5,861.86 | **0.5211** | Cân bằng vùng đầm nuôi thủy sản |
| **4** | **Rừng ngập mặn** | **140,251** | **6.08%** | **1,402.51** | **2.1781** | **Có mẫu lớn, trọng số tăng cường $\times 2.18$** |
| **5** | Cây lâu năm & Thực vật khác | 214,820 | 9.32% | 2,148.20 | **1.4220** | Cân bằng thảm thực vật / cây trồng |
| **6** | Đồng muối / Đất trống | 40,239 | 1.75% | 402.39 | **7.5917** | Tăng phạt mạnh lớp thiểu số |
| **Tổng** | | **2,305,200** | **100.00%** | **23,052.00** | | |

> [!NOTE]
> **Đánh giá chất lượng dữ liệu mới:**
> - Rừng ngập mặn (Lớp 4) từ $0\%$ đã đạt **$140,251$ pixel** (hơn $1,400$ ha), chiếm **$6.08\%$** tổng diện tích.
> - Cả 6 lớp đối tượng đều có số lượng pixel phong phú ($> 40,000$ pixel mỗi lớp), đảm bảo mô hình U-Net/ResNet50 ở Giai đoạn 2 học được đầy đủ đặc trưng phổ riêng biệt mà không bị suy biến gradient.

---

## 3. Biểu đồ Phân bố Cập nhật

![Phân bố diện tích 7 lớp sau khi mở rộng AOI](C:/Users/user/.gemini/antigravity-ide/brain/f05e09f0-c3a4-49c1-90d6-29cea7a75387/eda_class_distribution.png)

Các file kết xuất tại [data/processed/](file:///d:/project-TTDN/data/processed/):
- **Trọng số PyTorch**: [data/processed/class_weights.json](file:///d:/project-TTDN/data/processed/class_weights.json)
- **Tóm tắt EDA**: [data/processed/eda_summary.json](file:///d:/project-TTDN/data/processed/eda_summary.json)
- **Biểu đồ phân bố 7 lớp**: [data/processed/eda_class_distribution.png](file:///d:/project-TTDN/data/processed/eda_class_distribution.png)
- **Ảnh đối chiếu không gian**: [data/processed/eda_spatial_overlay.png](file:///d:/project-TTDN/data/processed/eda_spatial_overlay.png)
- **Mask 7 lớp mới**: [data/processed/giao_thuy_expanded_mask7.tif](file:///d:/project-TTDN/data/processed/giao_thuy_expanded_mask7.tif)

---

## 4. Kiểm thử Tự động Toàn diện

Bộ 8 unit tests đều tiếp tục đạt kết quả **100% PASSED**:
```text
Ran 8 tests in 2.774s - OK
```

---

## 5. Sẵn sàng Chuyển tiếp sang Bước 1.4

Tập dữ liệu mở rộng hiện đã hoàn hảo với đầy đủ 7 lớp. Chúng ta sẵn sàng bước sang **Bước 1.4: Tách dữ liệu theo không gian (Spatial Block Split 70% Train / 30% Golden Test)** trên cặp raster [giao_thuy_expanded_s2.tif](file:///d:/project-TTDN/data/raw/giao_thuy_expanded_s2.tif) và [giao_thuy_expanded_mask7.tif](file:///d:/project-TTDN/data/processed/giao_thuy_expanded_mask7.tif).
