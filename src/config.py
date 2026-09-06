from pathlib import Path
from typing import List, Tuple
import os

# Project directories (can be overridden via env vars)
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.getenv("PROJECT_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Sentinel-2 composite
GEE_DEFAULT_BANDS: List[str] = ["B2", "B3", "B4", "B8"]
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
GEE_CRS = "EPSG:32648"
GEE_SCALE_M = 10
GEE_CLOUD_PCT_MAX = 40
GEE_MAX_PIXELS = 10_000_000_000
# SCL classes excluded: no data, saturated, cloud shadow, cloud med/high, cirrus
SCL_EXCLUDE: Tuple[int, ...] = (0, 1, 3, 8, 9, 10)

# ESA WorldCover v200 (10 m, band Map)
WORLDCOVER_COLLECTION = "ESA/WorldCover/v200"
WORLDCOVER_BAND = "Map"

# ESA Map codes -> training classes 0-6 (Bước 1.2)
ESA_TO_LOCAL_CLASS: dict[int, int] = {
    40: 1,  # Lúa
    50: 2,  # Khu dân cư
    80: 3,  # Thủy sản
    95: 4,  # Rừng ngập mặn
    10: 5,  # Cây lâu năm & Thực vật khác (Tree cover)
    20: 5,  # Cây lâu năm & Thực vật khác (Shrubland)
    30: 5,  # Cây lâu năm & Thực vật khác (Grassland)
    60: 6,  # Đồng muối / Đất trống
}

CLASS_NAMES: dict[int, str] = {
    0: "Background",
    1: "Lúa",
    2: "Khu dân cư",
    3: "Thủy sản",
    4: "Rừng ngập mặn",
    5: "Cây lâu năm & Thực vật khác",
    6: "Đồng muối / Đất trống",
}

CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),        # Background: Đen / Nodata
    1: (255, 215, 0),    # Lúa: Vàng lúa
    2: (220, 20, 60),    # Khu dân cư: Đỏ đô
    3: (0, 191, 255),    # Thủy sản: Xanh nước biển
    4: (0, 100, 0),      # Rừng ngập mặn: Xanh lá đậm
    5: (34, 139, 34),    # Cây lâu năm & TV khác: Xanh rừng
    6: (210, 180, 140),  # Đồng muối / Đất trống: Nâu cát
}


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "INTERIM_DIR",
    "PROCESSED_DIR",
    "GEE_DEFAULT_BANDS",
    "S2_COLLECTION",
    "GEE_CRS",
    "GEE_SCALE_M",
    "GEE_CLOUD_PCT_MAX",
    "GEE_MAX_PIXELS",
    "SCL_EXCLUDE",
    "WORLDCOVER_COLLECTION",
    "WORLDCOVER_BAND",
    "ESA_TO_LOCAL_CLASS",
    "CLASS_NAMES",
    "CLASS_COLORS",
    "ensure_dirs",
]
