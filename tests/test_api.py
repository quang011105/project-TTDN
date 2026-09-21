"""
Kiểm thử tự động cho hệ thống API WebGIS Giám sát Quy hoạch Đất đai (FastAPI).
"""

import unittest
from fastapi.testclient import TestClient

from src.api.app import app

class TestWebGISApi(unittest.TestCase):
    """Bộ kiểm thử các endpoint API của WebGIS."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_administrative_units(self):
        """Kiểm tra API danh sách đơn vị hành chính."""
        response = self.client.get("/api/administrative-units")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        units = data["data"]
        self.assertGreaterEqual(len(units), 5)
        
        # Kiểm tra tồn tại huyện Giao Thủy và xã Bạch Long
        unit_ids = [u["id"] for u in units]
        self.assertIn("giao_thuy", unit_ids)
        self.assertIn("bach_long", unit_ids)
        self.assertIn("giao_thien", unit_ids)

    def test_get_single_unit(self):
        """Kiểm tra API lấy chi tiết một đơn vị."""
        response = self.client.get("/api/administrative-units/bach_long")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["data"]["id"], "bach_long")
        self.assertIn("bbox", data["data"])

    def test_lulc_stats_district(self):
        """Kiểm tra API thống kê diện tích toàn huyện."""
        response = self.client.get("/api/lulc-stats?unit_id=giao_thuy")
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(data["unit"]["id"], "giao_thuy")
        self.assertGreater(data["total_area_ha"], 22000.0)
        self.assertEqual(len(data["classes"]), 6)
        
        # Kiểm tra các lớp đất
        class_ids = [c["class_id"] for c in data["classes"]]
        self.assertEqual(class_ids, [1, 2, 3, 4, 5, 6])

    def test_lulc_stats_commune(self):
        """Kiểm tra API thống kê diện tích một xã cụ thể (Bạch Long)."""
        response = self.client.get("/api/lulc-stats?unit_id=bach_long")
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(data["unit"]["id"], "bach_long")
        self.assertGreater(data["total_pixels"], 0)
        self.assertEqual(len(data["classes"]), 6)

    def test_inspect_point_inside(self):
        """Kiểm tra API tra cứu pixel bên trong huyện Giao Thủy."""
        # Tọa độ trung tâm Giao Thủy
        response = self.client.get("/api/inspect?lat=20.250&lon=106.467")
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertTrue(data["inside"])
        self.assertIn("class_id", data)
        self.assertIn("class_name", data)
        self.assertIn("color", data)

    def test_inspect_point_outside(self):
        """Kiểm tra API tra cứu pixel ngoài phạm vi khảo sát (ở xa)."""
        response = self.client.get("/api/inspect?lat=10.0&lon=105.0")
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertFalse(data["inside"])
        self.assertEqual(data["class_id"], 0)

    def test_map_metadata(self):
        """Kiểm tra API thông số bounds và layers của bản đồ."""
        response = self.client.get("/api/map-metadata")
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertIn("leaflet_bounds", data)
        self.assertIn("layers", data)
        self.assertIn("lulc", data["layers"])

    def test_export_csv(self):
        """Kiểm tra API tải file CSV báo cáo diện tích."""
        response = self.client.get("/api/export-csv?unit_id=giao_thuy")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.headers["content-type"])
        content = response.content.decode("utf-8-sig")
        self.assertIn("BÁO CÁO HIỆN TRẠNG QUY HOẠCH SỬ DỤNG ĐẤT", content)
        self.assertIn("Lúa", content)
        self.assertIn("Thủy sản", content)


if __name__ == "__main__":
    unittest.main()
