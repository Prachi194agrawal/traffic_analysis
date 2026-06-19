import os
import sys
from pathlib import Path

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

import main


class FakePipeline:
    def analyze_image(self, image_path: str, conf: float, stopline_y_ratio: float):
        return {
            "annotated_name": "annotated_test.jpg",
            "summary": {
                "vehicle_count": 1,
                "plate_count": 0,
                "helmet_status": "helmet_unclear",
                "seatbelt_status": "not_checked",
                "red_signal": False,
                "green_signal": False,
                "yellow_signal": False,
                "crossed_vehicle_count": 0,
                "redlight_violation": False,
                "final_status": "no_clear_violation",
            },
            "meta": [
                {
                    "module": "vehicle_detection",
                    "class_name": "car",
                    "confidence": 0.9,
                    "bbox": [1, 2, 3, 4],
                    "ocr_text": None,
                    "ocr_confidence": None,
                    "rule": "vehicle_detected",
                    "status": "detected",
                }
            ],
        }


def test_health_and_analysis_history(monkeypatch):
    monkeypatch.setattr(main, "get_pipeline", lambda: FakePipeline())

    with TestClient(main.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "database": True}

        response = client.post(
            "/analyze",
            files={"file": ("traffic.jpg", b"fake-image", "image/jpeg")},
            data={"conf": "0.25", "stopline_y_ratio": "0.72"},
        )
        assert response.status_code == 200
        analysis_id = response.json()["analysis_id"]

        history = client.get("/analyses")
        assert history.status_code == 200
        assert history.json()["items"][0]["id"] == analysis_id

        detail = client.get(f"/analyses/{analysis_id}")
        assert detail.status_code == 200
        assert detail.json()["meta"][0]["class_name"] == "car"


def test_rejects_non_image_extension():
    with TestClient(main.app) as client:
        response = client.post(
            "/analyze",
            files={"file": ("payload.txt", b"not-an-image", "text/plain")},
        )
    assert response.status_code == 415
