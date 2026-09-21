"""
Định nghĩa các API endpoints cho hệ thống WebGIS.
"""

import json
from pathlib import Path
from fastapi import APIRouter, Query, HTTPException, Response
from fastapi.responses import PlainTextResponse

from src.api.administrative import get_all_units, get_unit_by_id
from src.api.services import get_lulc_statistics, inspect_point, generate_csv_report

router = APIRouter(prefix="/api", tags=["WebGIS API"])

METADATA_PATH = Path("web/assets/map_metadata.json")


@router.get("/administrative-units")
def list_administrative_units():
    """Lấy danh sách các đơn vị hành chính (Huyện và các Xã)."""
    return {
        "status": "success",
        "data": get_all_units()
    }


@router.get("/administrative-units/{unit_id}")
def get_administrative_unit(unit_id: str):
    """Lấy chi tiết một đơn vị hành chính theo ID."""
    unit = get_unit_by_id(unit_id)
    return {
        "status": "success",
        "data": unit
    }


@router.get("/lulc-stats")
def get_stats(unit_id: str = Query("giao_thuy", description="ID đơn vị hành chính")):
    """Lấy số liệu thống kê cơ cấu diện tích 6 loại đất theo đơn vị hành chính."""
    try:
        stats = get_lulc_statistics(unit_id)
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/inspect")
def inspect(
    lat: float = Query(..., description="Vĩ độ WGS84 (vd: 20.250)"),
    lon: float = Query(..., description="Kinh độ WGS84 (vd: 106.467)")
):
    """Tra cứu loại đất tại tọa độ kinh/vĩ độ (click chuột trên bản đồ)."""
    try:
        info = inspect_point(lat=lat, lon=lon)
        return {
            "status": "success",
            "data": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/map-metadata")
def get_map_metadata():
    """Lấy thông số Leaflet bounds, zoom và danh sách các lớp ảnh phủ."""
    if not METADATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Chưa khởi tạo map metadata.")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "status": "success",
        "data": meta
    }


@router.get("/export-csv")
def export_csv(unit_id: str = Query("giao_thuy")):
    """Tải xuống file CSV báo cáo diện tích quy hoạch đất đai."""
    try:
        csv_content = generate_csv_report(unit_id)
        filename = f"lulc_report_{unit_id}.csv"
        return Response(
            content=csv_content.encode("utf-8-sig"),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
