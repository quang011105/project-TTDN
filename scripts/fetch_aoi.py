"""Fetch AOI GeoJSON from Nominatim and save to disk.

Usage:
    python scripts/fetch_aoi.py --query "Giao Thủy Ninh Bình" --out data/raw/giao_thuy_ninh_binh.geojson
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


def fetch_nominatim_geojson(query: str, user_agent: str = "project-TTDN/0.1") -> dict:
    url = (
        "https://nominatim.openstreetmap.org/search.php"
        "?format=geojson&polygon_geojson=1&addressdetails=1&q=" + requests.utils.quote(query)
    )
    headers = {"User-Agent": user_agent}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gj = fetch_nominatim_geojson(args.query)

    # If multiple features, try to pick the feature whose display_name contains the province
    if gj.get("features"):
        # prefer feature with 'Ninh Bình' in display_name
        selected = None
        for feat in gj["features"]:
            dn = feat.get("properties", {}).get("display_name", "")
            if "Ninh Bình" in dn or "Ninh%20Binh" in dn:
                selected = feat
                break
        if selected:
            out = {"type": "FeatureCollection", "features": [selected]}
        else:
            out = gj
    else:
        out = gj

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"Saved AOI to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
