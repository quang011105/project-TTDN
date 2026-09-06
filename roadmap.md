Lộ trình phát triển hệ thống Giám sát Quy hoạch Sử dụng đất qua Viễn thám

Chiến lược: huấn luyện mô hình segmentation nhẹ (U-Net + ResNet50) trên cấp xã, dùng ESA WorldCover v200 làm Ground Truth tự động; inference phân tán trên cấp tỉnh qua HDFS và Apache Spark. Không vẽ ranh giới thủ công, không mồi nhãn bằng SAM. Phạm vi khả thi: 12 tuần, 1 kỹ sư — từ Data Pipeline đến Web Dashboard.

Khung thời gian (1 kỹ sư)
- Tuần 1–2: Giai đoạn 1 — ETL GEE, ánh xạ nhãn, spatial split, DVC.
- Tuần 3–5: Giai đoạn 2 — U-Net/ResNet50, huấn luyện và đánh giá cấp xã.
- Tuần 6–8: Giai đoạn 3 — tiling, HDFS, PySpark inference cấp tỉnh, FastAPI.
- Tuần 9–10: Giai đoạn 4 — GIS overlay, cảnh báo vi phạm, change detection, báo cáo.
- Tuần 11: Giai đoạn 5 — dashboard bản đồ và quản lý cảnh báo.
- Tuần 12: Giai đoạn 6 lõi (logging, kiểm thử tối thiểu) + ổn định tích hợp. Các hạng mục MLOps nặng đánh dấu Tùy chọn/Mở rộng.

Giai đoạn 1: Proof of Concept (PoC) & Chuẩn bị Dữ liệu Không gian (Cấp Xã)
Mục tiêu: Xây dựng ETL đồng bộ ảnh Sentinel-2 và nhãn ESA WorldCover v200; ánh xạ 11 lớp toàn cầu về 7 lớp Việt Nam; tạo tập Golden Test không rò rỉ không gian.

Bước 1.1 - Trích xuất ảnh và nhãn từ GEE: Một kịch bản Google Earth Engine tải đồng thời (1) composite Sentinel-2 L2A đa phổ theo ranh giới GeoJSON xã — mask mây (QA60/SCL), `median()` trên 3–6 tháng mùa khô; (2) raster ESA WorldCover v200 clip cùng ROI, cùng CRS và cùng lưới pixel (align/resample về 10 m). Xuất cặp GeoTIFF ảnh–nhãn.

Bước 1.2 - Ánh xạ nhãn (Class Mapping): Dùng NumPy ép 11 mã ESA về ma trận 7 kênh (chỉ số 0–6). Bảng ánh xạ bắt buộc: 40 → 1 (Lúa), 50 → 2 (Khu dân cư), 80 → 3 (Thủy sản), 95 → 4 (Rừng ngập mặn), 10 → 5 (Cây lâu năm), 60 → 6 (Đồng muối/Đất trống). Các mã còn lại (20, 30, 70, 90, 100) gộp vào Lớp nền (Background = 0). Lưu mask đã remap vào `data/processed`.

Bước 1.3 - Phân tích Khám phá Dữ liệu (EDA): Thống kê diện tích và tần suất 7 lớp sau remap. Heatmap phân bố lớp trên bản đồ nền. Tính class weights (Median Frequency Balancing) phục vụ Giai đoạn 2; Background (0) không tham gia cân bằng học.

Bước 1.4 - Tách dữ liệu theo không gian: 70% train / 30% Golden Test theo khối địa lý liền mạch (Spatial Block Split), không tách ngẫu nhiên theo pixel. Tránh spatial data leakage do autocorrelation. Spatial K-Fold chỉ khi cần CV trên tập train.

Bước 1.5 - Quản lý phiên bản dữ liệu: DVC theo dõi `data/raw` (Sentinel-2 + WorldCover gốc) và `data/processed` (mask 7 lớp, split). Metadata `.dvc` đi cùng Git để tái lập thí nghiệm.

Giai đoạn 2: Kiến trúc mạng Segmentation (Thị giác Máy tính)
Mục tiêu: Mô hình nhẹ, hội tụ nhanh trên cấp xã — tối ưu thuật toán, phần cứng và vòng lặp gỡ lỗi. Phạm vi huấn luyện chỉ giới hạn diện tích một xã.

Bước 2.1 - Thiết lập Backbone: U-Net với encoder ResNet50. Lớp đầu ra chốt cứng 7 channels (softmax/logits tương ứng lớp 0–6).

Bước 2.2 - Học chuyển giao: Khởi tạo encoder ImageNet (hoặc pre-train LandCover.ai / AgriFieldNet nếu thời gian cho phép). Mục tiêu: encoder học cấu trúc không gian; decoder học phổ Sentinel-2 + schema 7 lớp.

Bước 2.3 - Data Augmentation: `albumentations.Compose` đồng bộ ảnh và mask. Hình học: xoay 0°/90°/180°/270°, flip, random crop & resize. Phổ: color jitter, Gaussian noise, band dropout. Không gian: elastic / grid distortion mức vừa phải.

Bước 2.4 - Tinh chỉnh trên cấp xã: Freeze tầng sâu ResNet50 giai đoạn đầu; cập nhật Decoder (và unfreeze dần encoder). Chỉ dùng 70% diện tích xã đã split. Không huấn luyện trên raster toàn tỉnh.

Bước 2.5 - Hàm mất mát: Combo Loss = Weighted Cross-Entropy + Dice (+ Boundary nếu ổn định). Cross-Entropy bắt buộc `ignore_index=0` — không học lớp nền/nhiễu. Class weights chỉ trên lớp 1–6, tỉ lệ nghịch tần suất. Lovász-Softmax là tùy chọn nếu mIoU lớp hiếm chưa đạt.

Bước 2.6 - Chiến lược huấn luyện: AdamW (weight decay 1e-4), Cosine Annealing hoặc OneCycleLR, Mixed Precision FP16, Early Stopping theo validation mIoU (tính trên lớp 1–6). Gradient accumulation nếu VRAM hạn chế.

Bước 2.7 - Theo dõi thí nghiệm: MLflow hoặc Weights & Biases — metrics, hyperparameters, checkpoint. Ưu tiên một công cụ, cấu hình tối thiểu.

Bước 2.8 - Hậu xử lý: Morphological opening/closing, lọc polygon dưới ngưỡng diện tích, Douglas-Peucker. CRF đánh dấu Tùy chọn/Mở rộng (chi phí suy luận cao).

Bước 2.9 - Đánh giá: Golden Test 30% xã. Confusion matrix, Overall Accuracy, mIoU theo lớp 1–6 (bỏ Background), Kappa, Boundary F1. Không dùng pixel lớp 0 khi tính mIoU chính.

Giai đoạn 3: Kỹ thuật Hệ thống & Mở rộng (Cấp Tỉnh)
Mục tiêu: Sau khi mô hình hội tụ cấp xã, chạy phân vùng trên toàn tỉnh. Lưu hàng vạn patch trên HDFS; PySpark phân phối inference và thống kê diện tích song song.

Bước 3.1 - Đóng gói môi trường: Docker cho worker inference (PyTorch/ONNX + GDAL/Rasterio) và cho Spark job. `docker-compose.yml` cho các service lõi (API, worker); cluster Spark/HDFS dùng cấu hình riêng.

Bước 3.2 - Tối ưu Inference: Export ONNX Runtime (ưu tiên 12 tuần). FP16 mặc định; INT8/TensorRT là Tùy chọn/Mở rộng. Mục tiêu: giảm latency rõ so với PyTorch native mà giữ mIoU.

Bước 3.3 - Tiling & Stitching: Cắt raster tỉnh thành patch 256×256, overlap 32–64 px, ghi từng mảnh lên HDFS (kèm geotransform). Stitch bằng linear/Gaussian blend trên overlap; GDAL VRT hoặc catalog Spark để quản lý tile index.

Bước 3.4 - Phân phối trên HDFS + PySpark: HDFS lưu patches và mask dự đoán. PySpark: RDD/DataFrame các URI tile → mapPartitions gọi mô hình (broadcast weights hoặc UDF/pandas UDF) → ghi raster/parquet kết quả. Cùng cluster: reduce/aggregate diện tích theo lớp và theo đơn vị hành chính. Kiến trúc fan-out: driver chia partition → executor infer → collector stitch.

Bước 3.5 - Model Serving API: FastAPI — batch (danh sách tile/AOI huyện) và single-tile. Swagger cho frontend. Job tỉnh chạy Spark, không chặn request HTTP dài.

Bước 3.6 - Lịch cập nhật định kỳ: Cron/Airflow tối thiểu: hàng tháng kéo composite GEE cấp tỉnh → Spark infer. Apache Kafka streaming thời gian thực đánh dấu Tùy chọn/Mở rộng.

Giai đoạn 4: Đối chiếu Dữ liệu GIS & Phát hiện Thay đổi
Mục tiêu: Đưa raster 7 lớp cấp tỉnh vào quy trình hành chính: chồng quy hoạch, gắn cờ sai lệch, biến động theo thời gian.

Bước 4.1 - Chồng lớp GIS: Raster → vector (polygonize, simplify). Overlay với quy hoạch sử dụng đất tỉnh (cùng CRS).

Bước 4.2 - Cảnh báo vi phạm: Flag pixel/vùng hiện trạng (mô hình) ≠ pháp lý (quy hoạch), ví dụ quy hoạch lúa / mô hình khu dân cư. Ngưỡng diện tích tối thiểu để giảm false alarm.

Bước 4.3 - Phát hiện thay đổi: So sánh T1 vs T2 trên cùng lưới. Phân nhóm hợp pháp (khớp điều chỉnh quy hoạch) và bất hợp pháp (vi phạm mới).

Bước 4.4 - Báo cáo tự động: PDF/Excel — diện tích theo 7 lớp, danh sách vi phạm (tọa độ, ảnh minh họa), bản đồ biến động.

Giai đoạn 5: Dashboard & Giao diện Người dùng
Mục tiêu: Giao diện đủ dùng cho cán bộ đất đai trong tuần 11. Ưu tiên Streamlit hoặc Dash + Leaflet; React full-stack là Tùy chọn/Mở rộng.

Bước 5.1 - Web Dashboard: Bản đồ kết quả phân loại cấp xã/huyện/tỉnh. Lọc đơn vị hành chính, loại đất (1–6), loại vi phạm.

Bước 5.2 - Bản đồ so sánh: Before-after swipe T1/T2. Timeline mốc thời gian nếu đã có ≥2 kỳ infer.

Bước 5.3 - Quản lý cảnh báo: Bảng lọc/sắp xếp/xuất Excel. Trạng thái: pending → confirmed → resolved.

Giai đoạn 6: MLOps & Vận hành (lõi 12 tuần)
Mục tiêu: Hệ thống chạy ổn định, quan sát được. Không xây MLOps đầy đủ production trong 12 tuần.

Bước 6.1 - CI/CD Pipeline: Tùy chọn/Mở rộng. Trong 12 tuần: lint + test khi push (GitHub Actions tối thiểu) là đủ; không bắt buộc staging → production đa môi trường.

Bước 6.2 - Giám sát mô hình: So sánh phân bố lớp dự đoán giữa các kỳ infer (drift thô). Cảnh báo thủ công khi tỉ lệ lớp lệch mạnh. Data-drift platform đầy đủ: Tùy chọn/Mở rộng.

Bước 6.3 - Tái huấn luyện tự động: Tùy chọn/Mở rộng. Quy trình 12 tuần: retrain thủ công khi WorldCover/Sentinel cập nhật hoặc mIoU Golden Test giảm.

Bước 6.4 - Logging & quan sát: Structured logging toàn pipeline (GEE → remap → train → Spark job). Cảnh báo email khi job fail. Grafana + Prometheus: Tùy chọn/Mở rộng; log file + Spark UI là mặc định.

Bước 6.5 - Kiểm thử: Unit test ánh xạ nhãn, spatial split, ignore_index=0. Integration test ETL xã (ảnh + WorldCover → mask 7 lớp). Great Expectations và model regression tự động: Tùy chọn/Mở rộng.
