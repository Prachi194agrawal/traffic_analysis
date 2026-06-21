from pathlib import Path
from uuid import uuid4
import os
import re
from threading import RLock
from typing import Any, Dict, List, Tuple

import cv2
import easyocr
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import torch

from analysis_modules import ANALYSIS_MODULES
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
        self._models: Dict[str, Any] = {}
        self._reader = None
        self._load_lock = RLock()

    def get_model(self, model_key: str):
        with self._load_lock:
            if model_key not in self._models:
                self._models[model_key] = load_yolo_model(model_key)
            return self._models[model_key]

    def get_reader(self):
        with self._load_lock:
            if self._reader is None:
                print("[INFO] Loading EasyOCR...")
                self._reader = easyocr.Reader(
                    ["en"],
                    gpu=torch.cuda.is_available(),
                    model_storage_directory=str(EASYOCR_MODEL_DIR),
                )
                print("[INFO] EasyOCR loaded.")
            return self._reader

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
            results = self.get_reader().readtext(
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
    def analyze_image(
        self,
        image_path: str,
        conf: float = 0.25,
        stopline_y_ratio: float = 0.72,
        selected_modules: List[str] | None = None,
    ):
        selected_modules = selected_modules or list(ANALYSIS_MODULES)
        selected = set(selected_modules)
        img = self.read_image_any(image_path)
        if img is None:
            raise ValueError("Image not readable")

        annotated = img.copy()
        h, w = img.shape[:2]
        rows: List[Dict[str, Any]] = []
        module_results: Dict[str, Dict[str, Any]] = {}

        # Vehicle detections are also an internal dependency for seat belt crops.
        needs_vehicle_detections = "vehicle" in selected or "seatbelt" in selected
        vehicle_model = self.get_model("vehicle") if needs_vehicle_detections else None
        vehicle_dets_raw = self.predict_yolo(vehicle_model, img, conf=conf)
        vehicle_dets = []
        for det in vehicle_dets_raw:
            cname = det["class_name"].lower()
            if self.is_vehicle_class(cname):
                vehicle_dets.append(det)

        if "vehicle" in selected:
            for det in vehicle_dets:
                self.draw_box(
                    annotated,
                    det["bbox"],
                    f"{det['class_name'].title()}  {det['confidence']:.0%}",
                    (240, 126, 28),
                )
                rows.append({
                    "image_path": str(image_path), "module": "vehicle_detection", "class_name": det["class_name"],
                    "confidence": det["confidence"], "bbox": det["bbox"], "ocr_text": None, "ocr_confidence": None,
                    "rule": "vehicle_detected", "status": "detected"
                })
            module_results["vehicle"] = {
                "status": "complete" if vehicle_model is not None else "unavailable",
                "detections": len(vehicle_dets),
                "message": (
                    f"{len(vehicle_dets)} road vehicle{'s' if len(vehicle_dets) != 1 else ''} detected."
                    if vehicle_model is not None
                    else "Vehicle model is unavailable."
                ),
            }

        # License plate detection and OCR.
        plate_dets: List[Dict[str, Any]] = []
        recognized_plates: List[str] = []
        if "license_plate" in selected:
            plate_model = self.get_model("license_plate")
            plate_dets = self.predict_yolo(plate_model, img, conf=conf)
            for det in plate_dets:
                crop = self.crop_box(img, det["bbox"], pad=4)
                plate_text, ocr_conf = self.ocr_plate(crop)
                if plate_text:
                    recognized_plates.append(plate_text)
                self.draw_box(
                    annotated,
                    det["bbox"],
                    f"Plate  {plate_text or 'Text unclear'}",
                    (34, 211, 238),
                )
                rows.append({
                    "image_path": str(image_path), "module": "license_plate_ocr", "class_name": det["class_name"],
                    "confidence": det["confidence"], "bbox": det["bbox"], "ocr_text": plate_text,
                    "ocr_confidence": ocr_conf, "rule": "license_plate_detection",
                    "status": "recognized" if plate_text else "detected_text_unclear"
                })
            module_results["license_plate"] = {
                "status": "complete" if plate_model is not None else "unavailable",
                "detections": len(plate_dets),
                "recognized_values": recognized_plates,
                "message": (
                    f"{len(plate_dets)} plate{'s' if len(plate_dets) != 1 else ''} detected; "
                    f"{len(recognized_plates)} successfully read."
                    if plate_model is not None
                    else "License plate model is unavailable."
                ),
            }

        # Helmet compliance.
        helmet_dets: List[Dict[str, Any]] = []
        helmet_violation = False
        helmet_ok = False
        helmet_status = "not_selected"
        if "helmet" in selected:
            helmet_model = self.get_model("helmet")
            helmet_dets = self.predict_yolo(helmet_model, img, conf=conf)
            for det in helmet_dets:
                cname = det["class_name"]
                if self.is_no_helmet(cname):
                    helmet_violation = True
                    color, status, label = (38, 38, 220), "violation", f"Helmet required  {det['confidence']:.0%}"
                elif self.is_good_helmet(cname):
                    helmet_ok = True
                    color, status, label = (94, 183, 39), "compliant", f"Helmet compliant  {det['confidence']:.0%}"
                else:
                    color, status, label = (180, 180, 180), "review_required", f"{cname.replace('_', ' ').title()}  {det['confidence']:.0%}"
                self.draw_box(annotated, det["bbox"], label, color)
                rows.append({
                    "image_path": str(image_path), "module": "helmet_detection", "class_name": cname,
                    "confidence": det["confidence"], "bbox": det["bbox"], "ocr_text": None, "ocr_confidence": None,
                    "rule": "helmet_compliance", "status": status
                })
            if helmet_model is None:
                helmet_status = "model_unavailable"
            elif helmet_violation:
                helmet_status = "violation_detected"
            elif helmet_ok:
                helmet_status = "compliant"
            else:
                helmet_status = "not_detected"
            module_results["helmet"] = {
                "status": "complete" if helmet_model is not None else "unavailable",
                "detections": len(helmet_dets),
                "assessment": helmet_status,
                "message": {
                    "violation_detected": "At least one rider may not be wearing a helmet.",
                    "compliant": "Detected riders appear to be wearing helmets.",
                    "not_detected": "No helmet-related objects were detected.",
                    "model_unavailable": "Helmet model is unavailable.",
                }[helmet_status],
            }

        # Seat belt compliance on the upper region of supported vehicles.
        seatbelt_global_status = "not_selected"
        seatbelt_detection_count = 0
        if "seatbelt" in selected:
            seatbelt_model = self.get_model("seatbelt")
            supported_vehicles = [
                det for det in vehicle_dets
                if any(name in det["class_name"].lower() for name in ["car", "truck", "bus"])
            ]
            seatbelt_violation = False
            seatbelt_compliant = False
            review_required = False
            if seatbelt_model is not None:
                for vdet in supported_vehicles:
                    x1, y1, x2, y2 = vdet["bbox"]
                    crop_y2 = y1 + int((y2 - y1) * 0.55)
                    driver_crop = img[y1:crop_y2, x1:x2]
                    seatbelt_dets_crop = self.predict_yolo(seatbelt_model, driver_crop, conf=conf)
                    if len(seatbelt_dets_crop) == 0:
                        review_required = True
                        self.draw_box(annotated, vdet["bbox"], "Seat belt not visible - review", (0, 147, 255))
                        rows.append({
                            "image_path": str(image_path), "module": "seatbelt_detection", "class_name": "none",
                            "confidence": 0.0, "bbox": vdet["bbox"], "ocr_text": None, "ocr_confidence": None,
                            "rule": "seatbelt_compliance", "status": "review_required"
                        })
                    else:
                        for sdet in seatbelt_dets_crop:
                            seatbelt_detection_count += 1
                            sx1, sy1, sx2, sy2 = sdet["bbox"]
                            global_bbox = [x1 + sx1, y1 + sy1, x1 + sx2, y1 + sy2]
                            sname = sdet["class_name"]
                            if self.is_no_seatbelt(sname):
                                seatbelt_violation = True
                                color, status, label = (38, 38, 220), "violation", f"Seat belt required  {sdet['confidence']:.0%}"
                            elif self.is_seatbelt(sname):
                                seatbelt_compliant = True
                                color, status, label = (94, 183, 39), "compliant", f"Seat belt compliant  {sdet['confidence']:.0%}"
                            else:
                                review_required = True
                                color, status, label = (180, 180, 180), "review_required", f"{sname.replace('_', ' ').title()}  {sdet['confidence']:.0%}"
                            self.draw_box(annotated, global_bbox, label, color)
                            rows.append({
                                "image_path": str(image_path), "module": "seatbelt_detection", "class_name": sname,
                                "confidence": sdet["confidence"], "bbox": global_bbox, "ocr_text": None, "ocr_confidence": None,
                                "rule": "seatbelt_compliance", "status": status
                            })
            if seatbelt_model is None:
                seatbelt_global_status = "model_unavailable"
            elif not supported_vehicles:
                seatbelt_global_status = "no_supported_vehicle"
            elif seatbelt_violation:
                seatbelt_global_status = "violation_detected"
            elif review_required:
                seatbelt_global_status = "review_required"
            elif seatbelt_compliant:
                seatbelt_global_status = "compliant"
            else:
                seatbelt_global_status = "not_detected"
            seatbelt_messages = {
                "model_unavailable": "Seat belt model is unavailable.",
                "no_supported_vehicle": "No car, truck, or bus was available for seat belt review.",
                "violation_detected": "A possible seat belt violation was detected.",
                "review_required": "Seat belt use is not clear; manual review is recommended.",
                "compliant": "Detected seat belt use appears compliant.",
                "not_detected": "No seat belt-related objects were detected.",
            }
            module_results["seatbelt"] = {
                "status": "complete" if seatbelt_model is not None else "unavailable",
                "detections": seatbelt_detection_count,
                "vehicles_reviewed": len(supported_vehicles),
                "assessment": seatbelt_global_status,
                "message": seatbelt_messages[seatbelt_global_status],
            }

        # Traffic signal detection and optional stop-line rule.
        redlight_dets: List[Dict[str, Any]] = []
        red_signal = green_signal = yellow_signal = False
        traffic_signal_status = "not_selected"
        crossed_vehicle_count = None
        redlight_violation = None
        if "redlight" in selected:
            redlight_model = self.get_model("redlight")
            redlight_dets = self.predict_yolo(redlight_model, img, conf=conf)
            for det in redlight_dets:
                cname = det["class_name"]
                if self.is_red_light(cname):
                    red_signal = True
                    color, status, label = (38, 38, 220), "red_signal", f"Red signal  {det['confidence']:.0%}"
                elif self.is_green_light(cname):
                    green_signal = True
                    color, status, label = (94, 183, 39), "green_signal", f"Green signal  {det['confidence']:.0%}"
                elif self.is_yellow_light(cname):
                    yellow_signal = True
                    color, status, label = (0, 191, 255), "yellow_signal", f"Yellow signal  {det['confidence']:.0%}"
                else:
                    color, status, label = (180, 180, 180), "signal_detected", f"{cname.replace('_', ' ').title()}  {det['confidence']:.0%}"
                self.draw_box(annotated, det["bbox"], label, color)
                rows.append({
                    "image_path": str(image_path), "module": "redlight_detection", "class_name": cname,
                    "confidence": det["confidence"], "bbox": det["bbox"], "ocr_text": None, "ocr_confidence": None,
                    "rule": "traffic_signal_state", "status": status
                })

            if red_signal:
                traffic_signal_status = "red"
            elif green_signal:
                traffic_signal_status = "green"
            elif yellow_signal:
                traffic_signal_status = "yellow"
            elif redlight_model is None:
                traffic_signal_status = "model_unavailable"
            else:
                traffic_signal_status = "not_detected"

            rule_assessed = "vehicle" in selected and redlight_model is not None
            if rule_assessed:
                crossed_vehicle_count = 0
                stopline_y = int(h * stopline_y_ratio)
                cv2.line(annotated, (0, stopline_y), (w, stopline_y), (36, 62, 245), 2)
                cv2.putText(
                    annotated,
                    "STOP-LINE ASSESSMENT",
                    (20, max(30, stopline_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (36, 62, 245),
                    2,
                )
                for det in vehicle_dets:
                    if det["bbox"][3] > stopline_y:
                        crossed_vehicle_count += 1
                        rows.append({
                            "image_path": str(image_path), "module": "redlight_detection", "class_name": det["class_name"],
                            "confidence": det["confidence"], "bbox": det["bbox"], "ocr_text": None, "ocr_confidence": None,
                            "rule": "vehicle_crossed_stopline", "status": "stop_line_crossing"
                        })
                redlight_violation = bool(red_signal and crossed_vehicle_count > 0)

            if redlight_model is None:
                redlight_message = "Traffic signal model is unavailable."
            elif redlight_violation:
                redlight_message = "A vehicle crossed the configured stop line during a red signal."
            elif rule_assessed and red_signal:
                redlight_message = "A red signal was detected with no stop-line crossing."
            elif not rule_assessed and traffic_signal_status == "not_detected":
                redlight_message = "No traffic signal state was detected in this image."
            elif not rule_assessed:
                redlight_message = "Signal state detected. Enable Vehicle Detection to assess stop-line violations."
            else:
                redlight_message = f"Signal assessment complete: {traffic_signal_status.replace('_', ' ')}."
            module_results["redlight"] = {
                "status": "complete" if redlight_model is not None else "unavailable",
                "detections": len(redlight_dets),
                "signal": traffic_signal_status,
                "rule_assessed": rule_assessed,
                "crossed_vehicle_count": crossed_vehicle_count,
                "violation": redlight_violation,
                "message": redlight_message,
            }

        violation_found = (
            helmet_violation
            or redlight_violation is True
            or seatbelt_global_status == "violation_detected"
        )
        review_needed = seatbelt_global_status == "review_required"
        final_status = (
            "violation_detected"
            if violation_found
            else "review_required"
            if review_needed
            else "analysis_complete"
        )

        rows.append({
            "image_path": str(image_path), "module": "analysis_summary", "class_name": "result",
            "confidence": None, "bbox": None, "ocr_text": None, "ocr_confidence": None,
            "rule": "selected_module_assessment", "status": final_status
        })

        summary = {
            "image_path": str(image_path),
            "selected_modules": selected_modules,
            "vehicle_count": len(vehicle_dets) if "vehicle" in selected else None,
            "plate_count": len(plate_dets) if "license_plate" in selected else None,
            "recognized_plates": recognized_plates if "license_plate" in selected else None,
            "helmet_status": helmet_status,
            "seatbelt_status": seatbelt_global_status,
            "traffic_signal_status": traffic_signal_status,
            "red_signal": red_signal if "redlight" in selected else None,
            "green_signal": green_signal if "redlight" in selected else None,
            "yellow_signal": yellow_signal if "redlight" in selected else None,
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
            "selected_modules": selected_modules,
            "module_results": module_results,
        }
