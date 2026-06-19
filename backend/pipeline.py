from pathlib import Path
from uuid import uuid4
import os
import re
from typing import Any, Dict, List, Tuple

import cv2
import easyocr
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import torch

from model_manager import load_yolo_model

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EASYOCR_MODEL_DIR = Path(os.getenv("EASYOCR_MODEL_DIR", BASE_DIR / "models" / "easyocr"))
EASYOCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _to_builtin(value: Any):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    return value


def dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    df = df.where(pd.notnull(df), None)
    return [_to_builtin(row) for row in df.to_dict(orient="records")]


class TrafficPipeline:
    def __init__(self):
        self.vehicle_model = load_yolo_model("vehicle")
        self.license_plate_model = load_yolo_model("license_plate")
        self.helmet_model = load_yolo_model("helmet")
        self.seatbelt_model = load_yolo_model("seatbelt")
        self.redlight_model = load_yolo_model("redlight")

        print("[INFO] Loading EasyOCR...")
        self.reader = easyocr.Reader(
            ["en"],
            gpu=torch.cuda.is_available(),
            model_storage_directory=str(EASYOCR_MODEL_DIR),
        )
        print("[INFO] EasyOCR loaded.")

    # -------------------------
    # Reading + OCR
    # -------------------------
    def read_image_any(self, image_path: str):
        img = cv2.imread(str(image_path))
        if img is not None:
            return img
        pil_img = PILImage.open(image_path).convert("RGB")
        img = np.array(pil_img)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    @staticmethod
    def clean_text(text: str) -> str:
        text = str(text).upper()
        return re.sub(r"[^A-Z0-9]", "", text)

    def ocr_plate(self, crop) -> Tuple[str, float]:
        if crop is None or crop.size == 0:
            return "", 0.0
        try:
            crop = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            results = self.reader.readtext(
                gray,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )
            texts, confs = [], []
            for _, text, conf in results:
                cleaned = self.clean_text(text)
                if cleaned:
                    texts.append(cleaned)
                    confs.append(float(conf))
            if not texts:
                return "", 0.0
            return "".join(texts), float(np.mean(confs))
        except Exception as exc:
            print(f"[WARN] OCR skipped: {exc}")
            return "", 0.0

    # -------------------------
    # YOLO helpers
    # -------------------------
    @staticmethod
    def predict_yolo(model, img, conf: float = 0.25, imgsz: int = 640) -> List[Dict[str, Any]]:
        if model is None or img is None or getattr(img, "size", 0) == 0:
            return []
        try:
            results = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)
            if results is None or len(results) == 0:
                return []
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                return []

            detections = []
            for box in boxes:
                if box.cls is None or box.conf is None or box.xyxy is None:
                    continue
                cls_id = int(box.cls[0])
                score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                names = getattr(result, "names", None) or getattr(model, "names", {})
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": str(cls_name),
                        "confidence": score,
                        "bbox": [x1, y1, x2, y2],
                    }
                )
            return detections
        except Exception as exc:
            print(f"[WARN] Prediction skipped: {exc}")
            return []

    @staticmethod
    def crop_box(img, bbox, pad: int = 0):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2]

    @staticmethod
    def draw_box(img, bbox, label: str, color, thickness: int = 2):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        # background for readable labels
        label = str(label)[:40]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(18, y1 - 6)
        cv2.rectangle(img, (x1, y_text - th - 5), (x1 + tw + 5, y_text + 4), color, -1)
        cv2.putText(img, label, (x1 + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # -------------------------
    # Rule helper functions
    # -------------------------
    @staticmethod
    def is_vehicle_class(name: str) -> bool:
        name = str(name).lower()
        return any(w in name for w in ["car", "truck", "bus", "motorcycle", "bike", "bicycle", "auto", "vehicle"])

    @staticmethod
    def is_red_light(name: str) -> bool:
        return "red" in str(name).lower()

    @staticmethod
    def is_green_light(name: str) -> bool:
        return "green" in str(name).lower()

    @staticmethod
    def is_yellow_light(name: str) -> bool:
        name = str(name).lower()
        return "yellow" in name or "amber" in name

    @staticmethod
    def is_no_helmet(name: str) -> bool:
        name = str(name).lower()
        return any(w in name for w in ["no_helmet", "without_helmet", "nohelmet", "no-helmet", "no helmet", "head", "no_hardhat"])

    def is_good_helmet(self, name: str) -> bool:
        name = str(name).lower()
        if self.is_no_helmet(name):
            return False
        return any(w in name for w in ["helmet", "good_helmet", "with_helmet", "hardhat", "with helmet"])

    @staticmethod
    def is_no_seatbelt(name: str) -> bool:
        name = str(name).lower()
        return any(w in name for w in ["no_seatbelt", "no-seatbelt", "without_seatbelt", "without seatbelt", "no_belt", "no belt"])

    def is_seatbelt(self, name: str) -> bool:
        name = str(name).lower()
        if self.is_no_seatbelt(name):
            return False
        return "seatbelt" in name or "seat_belt" in name or "seat belt" in name or "belt" in name

    # -------------------------
    # Main pipeline
    # -------------------------
    def analyze_image(self, image_path: str, conf: float = 0.25, stopline_y_ratio: float = 0.72):
        img = self.read_image_any(image_path)
        if img is None:
            raise ValueError("Image not readable")

        annotated = img.copy()
        h, w = img.shape[:2]
        rows = []

        # 1. Vehicle detection
        vehicle_dets_raw = self.predict_yolo(self.vehicle_model, img, conf=conf)
        vehicle_dets = []
        for det in vehicle_dets_raw:
            cname = det["class_name"].lower()
            if self.is_vehicle_class(cname) or cname == "person":
                vehicle_dets.append(det)
                self.draw_box(annotated, det["bbox"], f"Vehicle {det['class_name']} {det['confidence']:.2f}", (255, 0, 0))
                rows.append({
                    "image_path": str(image_path), "module": "vehicle_detection", "class_name": det["class_name"],
                    "confidence": det["confidence"], "bbox": det["bbox"], "ocr_text": None, "ocr_confidence": None,
                    "rule": "vehicle_detected", "status": "detected"
                })

        # 2. License plate + OCR
        plate_dets = self.predict_yolo(self.license_plate_model, img, conf=conf)
        for det in plate_dets:
            crop = self.crop_box(img, det["bbox"], pad=4)
            plate_text, ocr_conf = self.ocr_plate(crop)
            self.draw_box(annotated, det["bbox"], f"Plate {plate_text or 'Unreadable'}", (0, 255, 255))
            rows.append({
                "image_path": str(image_path), "module": "license_plate_ocr", "class_name": det["class_name"],
                "confidence": det["confidence"], "bbox": det["bbox"], "ocr_text": plate_text,
                "ocr_confidence": ocr_conf, "rule": "license_plate_detection",
                "status": "plate_detected" if plate_text else "plate_detected_ocr_unreadable"
            })

        # 3. Helmet detection
        helmet_dets = self.predict_yolo(self.helmet_model, img, conf=conf)
        helmet_violation = False
        helmet_ok = False
        for det in helmet_dets:
            cname = det["class_name"]
            if self.is_no_helmet(cname):
                helmet_violation = True
                color, status, label = (0, 0, 255), "helmet_violation", f"No Helmet {det['confidence']:.2f}"
            elif self.is_good_helmet(cname):
                helmet_ok = True
                color, status, label = (0, 255, 0), "helmet_ok", f"Helmet {det['confidence']:.2f}"
            else:
                color, status, label = (255, 255, 255), "helmet_object_detected", f"{cname} {det['confidence']:.2f}"
            self.draw_box(annotated, det["bbox"], label, color)
            rows.append({
                "image_path": str(image_path), "module": "helmet_detection", "class_name": cname,
                "confidence": det["confidence"], "bbox": det["bbox"], "ocr_text": None, "ocr_confidence": None,
                "rule": "helmet_compliance", "status": status
            })

        # 4. Seatbelt detection on top crop of cars/buses/trucks
        seatbelt_global_status = "not_checked"
        if self.seatbelt_model is not None:
            for vdet in vehicle_dets:
                vname = vdet["class_name"].lower()
                if not any(x in vname for x in ["car", "truck", "bus"]):
                    continue
                x1, y1, x2, y2 = vdet["bbox"]
                crop_y2 = y1 + int((y2 - y1) * 0.55)
                driver_crop = img[y1:crop_y2, x1:x2]
                seatbelt_dets_crop = self.predict_yolo(self.seatbelt_model, driver_crop, conf=conf)
                if len(seatbelt_dets_crop) == 0:
                    seatbelt_global_status = "potential_seatbelt_violation"
                    self.draw_box(annotated, vdet["bbox"], "Seatbelt not detected", (0, 0, 255))
                    rows.append({
                        "image_path": str(image_path), "module": "seatbelt_detection", "class_name": "none",
                        "confidence": 0.0, "bbox": vdet["bbox"], "ocr_text": None, "ocr_confidence": None,
                        "rule": "seatbelt_compliance", "status": "potential_violation_no_seatbelt_detected"
                    })
                else:
                    for sdet in seatbelt_dets_crop:
                        sx1, sy1, sx2, sy2 = sdet["bbox"]
                        global_bbox = [x1 + sx1, y1 + sy1, x1 + sx2, y1 + sy2]
                        sname = sdet["class_name"]
                        if self.is_no_seatbelt(sname):
                            seatbelt_global_status = "seatbelt_violation"
                            color, status, label = (0, 0, 255), "seatbelt_violation", f"No Seatbelt {sdet['confidence']:.2f}"
                        elif self.is_seatbelt(sname):
                            seatbelt_global_status = "seatbelt_ok"
                            color, status, label = (0, 255, 0), "seatbelt_ok", f"Seatbelt {sdet['confidence']:.2f}"
                        else:
                            color, status, label = (255, 255, 255), "seatbelt_object_detected", f"{sname} {sdet['confidence']:.2f}"
                        self.draw_box(annotated, global_bbox, label, color)
                        rows.append({
                            "image_path": str(image_path), "module": "seatbelt_detection", "class_name": sname,
                            "confidence": sdet["confidence"], "bbox": global_bbox, "ocr_text": None, "ocr_confidence": None,
                            "rule": "seatbelt_compliance", "status": status
                        })

        # 5. Red light detection
        redlight_dets = self.predict_yolo(self.redlight_model, img, conf=conf)
        red_signal = green_signal = yellow_signal = False
        for det in redlight_dets:
            cname = det["class_name"]
            if self.is_red_light(cname):
                red_signal = True
                color, status, label = (0, 0, 255), "red_signal_detected", f"RED {det['confidence']:.2f}"
            elif self.is_green_light(cname):
                green_signal = True
                color, status, label = (0, 255, 0), "green_signal_detected", f"GREEN {det['confidence']:.2f}"
            elif self.is_yellow_light(cname):
                yellow_signal = True
                color, status, label = (0, 255, 255), "yellow_signal_detected", f"YELLOW {det['confidence']:.2f}"
            else:
                color, status, label = (255, 255, 255), "traffic_light_detected", f"{cname} {det['confidence']:.2f}"
            self.draw_box(annotated, det["bbox"], label, color)
            rows.append({
                "image_path": str(image_path), "module": "redlight_detection", "class_name": cname,
                "confidence": det["confidence"], "bbox": det["bbox"], "ocr_text": None, "ocr_confidence": None,
                "rule": "traffic_signal_state", "status": status
            })

        # 6. Stop-line heuristic
        stopline_y = int(h * stopline_y_ratio)
        cv2.line(annotated, (0, stopline_y), (w, stopline_y), (0, 0, 255), 2)
        cv2.putText(annotated, "STOP LINE ROI", (20, max(30, stopline_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        crossed_vehicle_count = 0
        for det in vehicle_dets:
            if det["bbox"][3] > stopline_y:
                crossed_vehicle_count += 1
                rows.append({
                    "image_path": str(image_path), "module": "stopline_check", "class_name": det["class_name"],
                    "confidence": det["confidence"], "bbox": det["bbox"], "ocr_text": None, "ocr_confidence": None,
                    "rule": "vehicle_crossed_stopline", "status": "crossed_stopline"
                })

        redlight_violation = bool(red_signal and crossed_vehicle_count > 0)
        if redlight_violation:
            redlight_status = "redlight_violation"
        elif red_signal:
            redlight_status = "red_signal_but_no_vehicle_crossed"
        else:
            redlight_status = "no_redlight_violation"

        rows.append({
            "image_path": str(image_path), "module": "final_rule_engine", "class_name": "summary",
            "confidence": None, "bbox": None, "ocr_text": None, "ocr_confidence": None,
            "rule": "redlight_violation_rule", "status": redlight_status
        })

        final_helmet_status = "helmet_violation" if helmet_violation else "helmet_ok" if helmet_ok else "helmet_unclear"
        final_status = "violation_found" if (
            helmet_violation or redlight_violation or seatbelt_global_status in ["seatbelt_violation", "potential_seatbelt_violation"]
        ) else "no_clear_violation"

        summary = {
            "image_path": str(image_path),
            "vehicle_count": len(vehicle_dets),
            "plate_count": len(plate_dets),
            "helmet_status": final_helmet_status,
            "seatbelt_status": seatbelt_global_status,
            "red_signal": red_signal,
            "green_signal": green_signal,
            "yellow_signal": yellow_signal,
            "crossed_vehicle_count": crossed_vehicle_count,
            "redlight_violation": redlight_violation,
            "final_status": final_status,
        }

        uid = uuid4().hex
        annotated_name = f"annotated_{uid}.jpg"
        meta_name = f"meta_{uid}.csv"
        summary_name = f"summary_{uid}.csv"
        annotated_path = OUTPUT_DIR / annotated_name
        meta_path = OUTPUT_DIR / meta_name
        summary_path = OUTPUT_DIR / summary_name

        cv2.imwrite(str(annotated_path), annotated)
        meta_df = pd.DataFrame(rows)
        summary_df = pd.DataFrame([summary])
        meta_df.to_csv(meta_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        return {
            "annotated_path": annotated_path,
            "annotated_name": annotated_name,
            "meta_path": meta_path,
            "summary_path": summary_path,
            "meta_df": meta_df,
            "summary_df": summary_df,
            "summary": summary,
            "meta": dataframe_to_records(meta_df),
        }
