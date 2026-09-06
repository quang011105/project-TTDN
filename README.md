# project-TTDN

Minimal project scaffold for the Phase 1 PoC.

Prerequisites
- Python 3.9+
- Earth Engine account and `earthengine` CLI authenticated

Install

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

Usage

Download aligned Sentinel-2 + ESA WorldCover v200 (UTM 10 m):

```bash
python -m src.gee_extractor path/to/aoi.geojson data/raw/giao_thuy --start 2024-01-01 --end 2024-03-31 --project YOUR_GCP_PROJECT
```

Writes `data/raw/giao_thuy_s2.tif` and `data/raw/giao_thuy_worldcover.tif`. For a large AOI use `--export-drive`.
