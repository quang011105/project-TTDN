"""
Ứng dụng FastAPI phục vụ WebGIS Giám sát Quy hoạch Sử dụng đất qua Viễn thám.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import router

app = FastAPI(
    title="Hệ thống Giám sát Quy hoạch Sử dụng đất qua Viễn thám",
    description="Hệ thống WebGIS thông minh tích hợp AI phân đoạn ngữ nghĩa ảnh vệ tinh Sentinel-2 huyện Giao Thủy.",
    version="1.0.0"
)

# Cấu hình CORS cho phép gọi API linh hoạt
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký các Router API
app.include_router(router)

# Mount thư mục Assets và Web Frontend
web_dir = Path("web")
assets_dir = web_dir / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)

app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)
