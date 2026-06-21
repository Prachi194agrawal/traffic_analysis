import os
import sys
from pathlib import Path

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

import main


class FakePipeline:
    def analyze_image(
        self,
        image_path: str,
        conf: float,
        stopline_y_ratio: float,
        selected_modules: list[str],
        preprocessing: str,
        no_parking_zone: list[float],
        legal_traffic_side: str,
    ):
        return {
            "annotated_name": "annotated_test.jpg",
            "selected_modules": selected_modules,
            "module_results": {
                key: {"status": "complete", "detections": 1, "message": "Test complete."}
                for key in selected_modules
            },
            "summary": {
                "selected_modules": selected_modules,
                "vehicle_count": 1,
                "plate_count": 0,
                "recognized_plates": [],
                "helmet_status": "not_detected",
                "seatbelt_status": "not_selected",
                "traffic_signal_status": "not_selected",
                "red_signal": False,
                "green_signal": False,
                "yellow_signal": False,
                "crossed_vehicle_count": None,
                "redlight_violation": None,
                "triple_riding_count": None,
                "wrong_side_review_count": None,
                "illegal_parking_review_count": None,
                "preprocessing": preprocessing,
                "risk_score": 0,
                "severity": "low",
                "decision_reasons": ["No clear violation was identified."],
                "recommendation": "Archive the analysis.",
                "final_status": "analysis_complete",
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
            "processed_name": None,
        }


def test_health_and_analysis_history(monkeypatch):
    monkeypatch.setattr(main, "get_pipeline", lambda: FakePipeline())

    with TestClient(main.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "database": True}

        modules = client.get("/modules")
        assert modules.status_code == 200
        assert len(modules.json()["items"]) == 8

        response = client.post(
            "/analyze",
            files={"file": ("traffic.jpg", b"fake-image", "image/jpeg")},
            data={
                "conf": "0.25",
                "stopline_y_ratio": "0.72",
                "modules": "vehicle,helmet",
            },
        )
        assert response.status_code == 200
        assert response.json()["selected_modules"] == ["vehicle", "helmet"]
        analysis_id = response.json()["analysis_id"]

        history = client.get("/analyses")
        assert history.status_code == 200
        assert history.json()["items"][0]["id"] == analysis_id

        detail = client.get(f"/analyses/{analysis_id}")
        assert detail.status_code == 200
        assert detail.json()["meta"][0]["class_name"] == "car"

        pdf = client.get(f"/analyses/{analysis_id}/report.pdf")
        assert pdf.status_code == 200
        assert pdf.headers["content-type"] == "application/pdf"
        assert pdf.content.startswith(b"%PDF")

        csv_report = client.get(f"/analyses/{analysis_id}/report.csv")
        assert csv_report.status_code == 200
        assert "vehicle_detection" in csv_report.text

        analytics = client.get("/analytics")
        assert analytics.status_code == 200
        assert analytics.json()["total_analyses"] >= 1

        evaluation = client.get("/evaluation")
        assert evaluation.status_code == 200
        assert len(evaluation.json()["models"]) == 5


def test_rejects_non_image_extension():
    with TestClient(main.app) as client:
        response = client.post(
            "/analyze",
            files={"file": ("payload.txt", b"not-an-image", "text/plain")},
        )
    assert response.status_code == 415


def test_rejects_unknown_analysis_module():
    with TestClient(main.app) as client:
        response = client.post(
            "/analyze",
            files={"file": ("traffic.jpg", b"fake-image", "image/jpeg")},
            data={"modules": "vehicle,unknown"},
        )
    assert response.status_code == 422
    assert "Unsupported analysis module" in response.json()["detail"]
