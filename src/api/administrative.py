"""
Danh mục Đơn vị Hành chính huyện Giao Thủy (thuộc tỉnh Nam Định / Ninh Bình mới)
Chứa thông tin tọa độ tâm, mức zoom và bounding box WGS84 phục vụ WebGIS flyTo.
"""

from typing import Dict, List, Any

ADMINISTRATIVE_UNITS: Dict[str, Dict[str, Any]] = {
    "giao_thuy": {
        "id": "giao_thuy",
        "name": "Toàn Huyện Giao Thủy",
        "type": "district",
        "parent": "Nam Định (Ninh Bình)",
        "center": [20.2502, 106.4679],
        "zoom": 12,
        "bbox": [
            [20.1983, 106.3698],
            [20.3020, 106.5661]
        ],
        "description": "Huyện ven biển vùng cực nam đồng bằng sông Hồng, tiếp giáp vịnh Bắc Bộ và cửa sông Ba Lạt (Vườn Quốc gia Xuân Thủy)."
    },
    "bach_long": {
        "id": "bach_long",
        "name": "Xã Bạch Long (Cánh đồng muối)",
        "type": "commune",
        "parent": "Huyện Giao Thủy",
        "center": [20.2150, 106.4250],
        "zoom": 14,
        "bbox": [
            [20.2000, 106.4100],
            [20.2300, 106.4400]
        ],
        "description": "Địa bàn có cánh đồng muối lớn nhất miền Bắc và vùng nuôi trồng thủy sản nước mặn ven đê biển."
    },
    "giao_thien": {
        "id": "giao_thien",
        "name": "Xã Giao Thiện (Cửa Ba Lạt - VQG Xuân Thủy)",
        "type": "commune",
        "parent": "Huyện Giao Thủy",
        "center": [20.2700, 106.5300],
        "zoom": 14,
        "bbox": [
            [20.2500, 106.5050],
            [20.2900, 106.5600]
        ],
        "description": "Vùng cửa sông Hồng (Ba Lạt) đặc trưng với thảm rừng ngập mặn phòng hộ và vùng đệm Ramsar Xuân Thủy."
    },
    "quat_lam": {
        "id": "quat_lam",
        "name": "Thị trấn Quất Lâm (Đô thị Biển)",
        "type": "town",
        "parent": "Huyện Giao Thủy",
        "center": [20.2050, 106.3750],
        "zoom": 14,
        "bbox": [
            [20.1983, 106.3698],
            [20.2200, 106.3950]
        ],
        "description": "Trung tâm dịch vụ, du lịch biển và khu dân cư đô thị hóa ở rìa Tây Nam huyện."
    },
    "ngo_dong": {
        "id": "ngo_dong",
        "name": "Thị trấn Ngô Đồng (Trung tâm Huyện lỵ)",
        "type": "town",
        "parent": "Huyện Giao Thủy",
        "center": [20.2850, 106.3950],
        "zoom": 14,
        "bbox": [
            [20.2700, 106.3800],
            [20.3000, 106.4100]
        ],
        "description": "Trung tâm chính trị, hành chính, văn hóa và mật độ dân cư tập trung cao nhất huyện."
    },
    "giao_an": {
        "id": "giao_an",
        "name": "Xã Giao An (Cửa ngõ Rừng ngập mặn)",
        "type": "commune",
        "parent": "Huyện Giao Thủy",
        "center": [20.2500, 106.5100],
        "zoom": 14,
        "bbox": [
            [20.2350, 106.4900],
            [20.2650, 106.5300]
        ],
        "description": "Xã nằm tiếp giáp VQG Xuân Thủy với diện tích nuôi trồng thủy sản nước lợ và rừng sú vẹt bạt ngàn."
    },
    "giao_lac": {
        "id": "giao_lac",
        "name": "Xã Giao Lạc (Vùng bãi triều & Đầm tôm)",
        "type": "commune",
        "parent": "Huyện Giao Thủy",
        "center": [20.2300, 106.4800],
        "zoom": 14,
        "bbox": [
            [20.2150, 106.4600],
            [20.2450, 106.5000]
        ],
        "description": "Khu vực kinh tế thủy sản trọng điểm bãi triều ven sông Sò và biển Đông."
    },
    "hoanh_son": {
        "id": "hoanh_son",
        "name": "Xã Hoành Sơn (Nội đồng trồng Lúa)",
        "type": "commune",
        "parent": "Huyện Giao Thủy",
        "center": [20.2750, 106.4150],
        "zoom": 14,
        "bbox": [
            [20.2600, 106.4000],
            [20.2900, 106.4300]
        ],
        "description": "Vùng thâm canh lúa nước truyền thống với các cụm dân cư nông thôn thuần hậu ven sông."
    },
    "giao_tien": {
        "id": "giao_tien",
        "name": "Xã Giao Tiến (Nông thôn trù phú)",
        "type": "commune",
        "parent": "Huyện Giao Thủy",
        "center": [20.2800, 106.3800],
        "zoom": 14,
        "bbox": [
            [20.2650, 106.3700],
            [20.2950, 106.3950]
        ],
        "description": "Xã phát triển làng nghề, nông nghiệp và khu dân cư trù phú phía Tây huyện."
    },
    "giao_xuan": {
        "id": "giao_xuan",
        "name": "Xã Giao Xuân (Vùng nuôi Ngao & Du lịch sinh thái)",
        "type": "commune",
        "parent": "Huyện Giao Thủy",
        "center": [20.2400, 106.4950],
        "zoom": 14,
        "bbox": [
            [20.2250, 106.4800],
            [20.2550, 106.5150]
        ],
        "description": "Vùng nuôi nghêu ngao bãi bồi nổi tiếng và mô hình du lịch sinh thái cộng đồng Ramsar."
    }
}

def get_all_units() -> List[Dict[str, Any]]:
    """Trả về danh sách đơn vị hành chính phục vụ dropdown chọn địa bàn."""
    return list(ADMINISTRATIVE_UNITS.values())

def get_unit_by_id(unit_id: str) -> Dict[str, Any]:
    """Tìm đơn vị hành chính theo ID, mặc định trả về toàn huyện."""
    return ADMINISTRATIVE_UNITS.get(unit_id, ADMINISTRATIVE_UNITS["giao_thuy"])
