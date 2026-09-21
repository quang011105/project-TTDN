"""
Các dịch vụ xử lý dữ liệu GIS và phân tích thống kê diện tích LULC phục vụ API.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import rasterio
from rasterio.warp import transform
from rasterio.windows import from_bounds

from src.config import CLASS_NAMES, CLASS_COLORS
from src.api.administrative import get_unit_by_id

LULC_TIF_PATH = Path("outputs/maps/giao_thuy_lulc_prediction.tif")
LULC_WGS84_PATH = Path("outputs/maps/giao_thuy_lulc_wgs84.tif")
STATS_JSON_PATH = Path("outputs/maps/lulc_area_statistics.json")

CLASS_DESCRIPTIONS: Dict[int, str] = {
    0: "Vùng nước biển / Ngoài ranh giới khảo sát (Nodata)",
    1: "Đất canh tác lúa nước hai vụ, vùng phù sa màu mỡ nội đồng",
    2: "Khu dân cư, nhà ở nông thôn, công trình xây dựng và hạ tầng giao thông",
    3: "Đầm nuôi tôm, cua, cá nước lợ và bãi triều nuôi ngao ven biển",
    4: "Thảm rừng ngập mặn (cây bần, trang, sú vẹt) VQG Xuân Thủy & rừng phòng hộ đê",
    5: "Cây lâu năm, vườn cây ăn quả quanh nhà, luỹ tre và thảm thực vật khác",
    6: "Ruộng phơi muối kết tinh (Bạch Long) và bãi cát bồi tự nhiên ven biển"
}

def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"

def get_lulc_statistics(unit_id: str = "giao_thuy") -> Dict[str, Any]:
    """Tính toán hoặc truy vấn số liệu thống kê LULC theo đơn vị hành chính."""
    unit = get_unit_by_id(unit_id)
    
    # 1. Nếu là toàn huyện, nạp file JSON thống kê đã có sẵn
    if unit_id == "giao_thuy" and STATS_JSON_PATH.exists():
        with open(STATS_JSON_PATH, "r", encoding="utf-8") as f:
            stats_data = json.load(f)

        stats_list = []
        classes_raw = stats_data.get("classes", stats_data.get("statistics", []))
        if isinstance(classes_raw, dict):
            classes_iter = classes_raw.values()
        else:
            classes_iter = classes_raw

        for row in classes_iter:
            cid = int(row["class_id"])
            rgb = CLASS_COLORS.get(cid, (128, 128, 128))
            stats_list.append({
                "class_id": cid,
                "name": row.get("class_name", CLASS_NAMES.get(cid, f"Lớp {cid}")),
                "pixels": int(row.get("pixel_count", 0)),
                "area_ha": float(row.get("area_ha", 0.0)),
                "area_km2": float(row.get("area_km2", 0.0)),
                "percentage": float(row.get("percentage", 0.0)),
                "color": rgb_to_hex(*rgb),
                "description": CLASS_DESCRIPTIONS.get(cid, "")
            })

        # Sắp xếp theo class_id
        stats_list.sort(key=lambda x: x["class_id"])

        total_px = stats_data.get("total_foreground_pixels", stats_data.get("total_valid_pixels", 2256537))
        return {
            "unit": unit,
            "total_pixels": int(total_px),
            "total_area_ha": float(stats_data.get("total_area_ha", 22565.37)),
            "total_area_km2": float(stats_data.get("total_area_km2", 225.65)),
            "classes": stats_list
        }

    # 2. Nếu là một xã cụ thể, cắt raster theo Bounding Box
    if not LULC_TIF_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy raster LULC: {LULC_TIF_PATH}")

    bbox = unit["bbox"] # [[lat_min, lon_min], [lat_max, lon_max]]
    lat_min, lon_min = bbox[0]
    lat_max, lon_max = bbox[1]

    with rasterio.open(LULC_TIF_PATH) as src:
        # Chuyển đổi 4 góc BBOX từ EPSG:4326 sang EPSG:32648
        xs, ys = transform("EPSG:4326", src.crs, [lon_min, lon_max], [lat_min, lat_max])
        left, right = min(xs), max(xs)
        bottom, top = min(ys), max(ys)

        window = from_bounds(left, bottom, right, top, transform=src.transform)
        # Giới hạn window trong phạm vi ảnh
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

        if window.width <= 0 or window.height <= 0:
            sub_mask = np.array([], dtype=np.uint8)
        else:
            sub_mask = src.read(1, window=window)

    # Đếm số pixel các lớp 1-6
    valid_mask = sub_mask[(sub_mask >= 1) & (sub_mask <= 6)]
    total_valid = len(valid_mask)
    if total_valid == 0:
        total_valid = 1 # Tránh chia cho 0

    classes_stats = []
    for cid in range(1, 7):
        px_count = int(np.sum(sub_mask == cid))
        area_ha = round(px_count * 0.01, 2)
        area_km2 = round(area_ha / 100.0, 2)
        pct = round((px_count / total_valid) * 100.0, 2)
        rgb = CLASS_COLORS.get(cid, (128, 128, 128))
        classes_stats.append({
            "class_id": cid,
            "name": CLASS_NAMES.get(cid, f"Lớp {cid}"),
            "pixels": px_count,
            "area_ha": area_ha,
            "area_km2": area_km2,
            "percentage": pct,
            "color": rgb_to_hex(*rgb),
            "description": CLASS_DESCRIPTIONS.get(cid, "")
        })

    tot_ha = round(total_valid * 0.01, 2)
    tot_km2 = round(tot_ha / 100.0, 2)

    return {
        "unit": unit,
        "total_pixels": total_valid,
        "total_area_ha": tot_ha,
        "total_area_km2": tot_km2,
        "classes": classes_stats
    }


def inspect_point(lat: float, lon: float) -> Dict[str, Any]:
    """Tra cứu loại đất tại một tọa độ kinh/vĩ độ (WGS84) bất kỳ."""
    target_tif = LULC_WGS84_PATH if LULC_WGS84_PATH.exists() else LULC_TIF_PATH
    if not target_tif.exists():
        raise FileNotFoundError("Không tìm thấy file raster LULC.")

    with rasterio.open(target_tif) as src:
        is_wgs84 = (str(src.crs).upper() == "EPSG:4326")
        if is_wgs84:
            x_proj, y_proj = lon, lat
        else:
            xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
            x_proj, y_proj = xs[0], ys[0]

        # Kiểm tra điểm có nằm trong bounds không
        if (x_proj < src.bounds.left or x_proj > src.bounds.right or
            y_proj < src.bounds.bottom or y_proj > src.bounds.top):
            return {
                "lat": lat,
                "lon": lon,
                "inside": False,
                "class_id": 0,
                "class_name": "Ngoài phạm vi khảo sát",
                "color": "#111827",
                "description": "Điểm nằm ngoài ranh giới huyện Giao Thủy hoặc trên biển."
            }

        # Lấy chỉ số hàng, cột pixel
        row, col = src.index(x_proj, y_proj)
        row = max(0, min(row, src.height - 1))
        col = max(0, min(col, src.width - 1))

        # Đọc giá trị pixel
        window = rasterio.windows.Window(col, row, 1, 1)
        val = int(src.read(1, window=window)[0, 0])

    c_name = CLASS_NAMES.get(val, "Chưa phân loại")
    rgb = CLASS_COLORS.get(val, (128, 128, 128))
    desc = CLASS_DESCRIPTIONS.get(val, "")

    return {
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "row": int(row),
        "col": int(col),
        "inside": True,
        "class_id": val,
        "class_name": c_name,
        "color": rgb_to_hex(*rgb),
        "description": desc
    }


def generate_csv_report(unit_id: str = "giao_thuy") -> str:
    """Tạo chuỗi CSV báo cáo thống kê diện tích."""
    data = get_lulc_statistics(unit_id)
    unit_name = data["unit"]["name"]

    lines = [
        f"BÁO CÁO HIỆN TRẠNG QUY HOẠCH SỬ DỤNG ĐẤT - {unit_name.upper()}",
        f"Đơn vị tính: Hécta (ha) và Kilômét vuông (km²)",
        "",
        "Mã lớp,Tên loại đất,Số pixel,Diện tích (ha),Diện tích (km²),Tỷ lệ (%)"
    ]

    for c in data["classes"]:
        lines.append(f"{c['class_id']},{c['name']},{c['pixels']},{c['area_ha']},{c['area_km2']},{c['percentage']}%")

    lines.append("")
    lines.append(f"TỔNG CỘNG,,{data['total_pixels']},{data['total_area_ha']},{data['total_area_km2']},100.00%")

    return "\n".join(lines)
