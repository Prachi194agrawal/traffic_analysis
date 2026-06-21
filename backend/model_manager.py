import os
import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, list_repo_files
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
BUNDLED_MODELS_DIR = Path(os.getenv("BUNDLED_MODELS_DIR", BASE_DIR / "bundled-models"))

MODEL_SOURCES = {
    "license_plate": {
        "local_file": "license_plate_best.pt",
        "repo_id": "Koushim/yolov8-license-plate-detection",
        "filename": "best.pt",
    },
    "helmet": {
        "local_file": "helmet_best.pt",
        "repo_id": "sharathhhhh/safetyHelmet-detection-yolov8",
        "filename": "best.pt",
    },
    "seatbelt": {
        "local_file": "seatbelt_best.pt",
        "repo_id": "RISEF/yolov11s-seatbelt",
        "filename": None,
    },
    "redlight": {
        "local_file": "redlight_best.pt",
        "repo_id": "mehmetkeremturkcan/traffic-lights-of-new-york",
        "filename": None,
    },
}


def _find_pt_file(repo_id: str) -> Optional[str]:
    files = list_repo_files(repo_id)
    pt_files = [f for f in files if f.endswith(".pt")]
    if not pt_files:
        return None
    best_files = [f for f in pt_files if Path(f).name == "best.pt"]
    return best_files[0] if best_files else pt_files[0]


def _download_hf_model(repo_id: str, target_path: Path, filename: Optional[str] = None) -> Optional[Path]:
    try:
        if filename is None:
            filename = _find_pt_file(repo_id)
            if filename is None:
                print(f"[WARN] No .pt model found in Hugging Face repo: {repo_id}")
                return None

        print(f"[INFO] Downloading {repo_id}/{filename}")
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copy2(downloaded, target_path)
        print(f"[INFO] Saved model to {target_path}")
        return target_path
    except Exception as exc:
        print(f"[WARN] Could not download {repo_id}: {exc}")
        return None


def ensure_model_file(model_key: str) -> Optional[Path]:
    if model_key == "vehicle":
        target = MODELS_DIR / "vehicle_best.pt"
        if target.exists():
            return target
        bundled_target = BUNDLED_MODELS_DIR / "vehicle_best.pt"
        if bundled_target.exists():
            return bundled_target
        try:
            # Ultralytics downloads yolov8n.pt automatically if not present.
            YOLO("yolov8n.pt")
            if Path("yolov8n.pt").exists():
                shutil.copy2("yolov8n.pt", target)
            else:
                # If not in cwd, still return the official name for YOLO loading.
                target = Path("yolov8n.pt")
            return target
        except Exception as exc:
            print(f"[WARN] Vehicle model unavailable: {exc}")
            return None

    cfg = MODEL_SOURCES.get(model_key)
    if cfg is None:
        return None

    target = MODELS_DIR / cfg["local_file"]
    if target.exists():
        return target

    bundled_target = BUNDLED_MODELS_DIR / cfg["local_file"]
    if bundled_target.exists():
        return bundled_target

    return _download_hf_model(
        repo_id=cfg["repo_id"],
        filename=cfg["filename"],
        target_path=target,
    )


def load_yolo_model(model_key: str):
    model_path = ensure_model_file(model_key)
    if model_path is None:
        print(f"[WARN] {model_key} model disabled because weights are missing.")
        return None

    try:
        model = YOLO(str(model_path))
        print(f"[INFO] Loaded {model_key}: {model_path}")
        print(f"[INFO] {model_key} classes: {getattr(model, 'names', {})}")
        return model
    except Exception as exc:
        print(f"[WARN] Could not load {model_key} model from {model_path}: {exc}")
        return None
