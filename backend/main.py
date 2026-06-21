import os
import time
import csv
from contextlib import asynccontextmanager
from functools import lru_cache
from io import StringIO
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from analysis_modules import ANALYSIS_MODULES, normalize_modules
from database import (
    analysis_to_dict,
    create_analysis,
    database_is_ready,
    get_analytics,
    get_analysis,
    init_database,
    list_analyses,
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", BASE_DIR / "static"))
OUTPUT_DIR = STATIC_DIR / "outputs"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PREPROCESSING_MODES = {"none", "auto", "low_light", "denoise", "sharpen", "contrast"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_pipeline():
    # ML libraries and model weights are intentionally loaded only on first analysis.
    from pipeline import TrafficPipeline

    return TrafficPipeline()


def run_analysis(
    image_path: str,
    conf: float,
    stopline_y_ratio: float,
    selected_modules: list[str],
    preprocessing: str,
    no_parking_zone: list[float],
    legal_traffic_side: str,
):
    return get_pipeline().analyze_image(
        image_path=image_path,
        conf=conf,
        stopline_y_ratio=stopline_y_ratio,
        selected_modules=selected_modules,
        preprocessing=preprocessing,
        no_parking_zone=no_parking_zone,
        legal_traffic_side=legal_traffic_side,
    )


def parse_zone(value: str) -> list[float]:
    try:
        zone = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("No-parking zone must contain four decimal coordinates") from exc
    if len(zone) != 4 or any(item < 0 or item > 1 for item in zone):
        raise ValueError("No-parking zone must contain four values between 0 and 1")
    if zone[0] >= zone[2] or zone[1] >= zone[3]:
        raise ValueError("No-parking zone must have positive width and height")
    return zone


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_database()
    yield


app = FastAPI(
    title="Traffic Analysis AI",
    version="1.0.0",
    lifespan=lifespan,
    root_path=os.getenv("ROOT_PATH", ""),
)

allowed_origins = [
    value.strip()
    for value in os.getenv("CORS_ORIGINS", "*").split(",")
    if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return {"message": "Traffic Analysis AI backend is running", "docs": "/docs"}


@app.get("/health")
def health():
    database_ok = database_is_ready()
    return JSONResponse(
        status_code=200 if database_ok else 503,
        content={"status": "ok" if database_ok else "degraded", "database": database_ok},
    )


@app.get("/modules")
def available_modules():
    return {"items": [{"key": key, **details} for key, details in ANALYSIS_MODULES.items()]}


@app.get("/analyses")
def analysis_history(limit: int = 20):
    safe_limit = max(1, min(limit, 100))
    return {"items": [analysis_to_dict(item) for item in list_analyses(safe_limit)]}


@app.get("/analytics")
def analytics_dashboard():
    return get_analytics()


@app.get("/evaluation")
def evaluation_dashboard():
    analytics = get_analytics()
    bundled_dir = Path(os.getenv("BUNDLED_MODELS_DIR", BASE_DIR / "bundled-models"))
    local_dir = BASE_DIR / "models"
    models = []
    for key in ["vehicle", "license_plate", "helmet", "seatbelt", "redlight"]:
        filename = f"{key}_best.pt"
        path = bundled_dir / filename
        if not path.exists():
            path = local_dir / filename
        models.append({
            "module": key,
            "model_available": path.exists(),
            "artifact_size_mb": round(path.stat().st_size / 1024 / 1024, 2) if path.exists() else None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "map50": None,
            "quality_status": "Ground-truth validation dataset required",
        })
    return {
        "runtime": {
            "average_processing_ms": analytics["average_processing_ms"],
            "measured_samples": analytics["measured_latency_samples"],
        },
        "models": models,
        "methodology": [
            "Runtime is measured end-to-end for each newly processed image.",
            "Precision, recall, F1 and mAP are intentionally not fabricated.",
            "Populate quality metrics after evaluation against a labeled, held-out dataset.",
        ],
    }


@app.get("/analyses/{analysis_id}")
def analysis_detail(analysis_id: str):
    analysis = get_analysis(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis_to_dict(analysis, include_detections=True)


@app.get("/analyses/{analysis_id}/report.pdf")
def analysis_pdf_report(analysis_id: str):
    analysis = get_analysis(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    from reports import build_pdf_report

    pdf = build_pdf_report(analysis, STATIC_DIR)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="roadsight-{analysis_id}.pdf"'},
    )


@app.get("/analyses/{analysis_id}/report.csv")
def analysis_csv_report(analysis_id: str):
    analysis = get_analysis(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["module", "class_name", "confidence", "status", "rule", "ocr_text", "bbox"])
    for item in analysis.detections:
        writer.writerow([
            item.module,
            item.class_name,
            item.confidence,
            item.status,
            item.rule,
            item.ocr_text,
            item.bbox,
        ])
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="roadsight-{analysis_id}.csv"'},
    )


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    stopline_y_ratio: float = Form(0.72),
    modules: str = Form("all"),
    preprocessing: str = Form("none"),
    no_parking_zone: str = Form("0.65,0.35,0.98,0.95"),
    legal_traffic_side: str = Form("left"),
):
    if not 0.0 < conf <= 1.0:
        raise HTTPException(status_code=422, detail="conf must be between 0 and 1")
    if not 0.0 < stopline_y_ratio < 1.0:
        raise HTTPException(status_code=422, detail="stopline_y_ratio must be between 0 and 1")
    try:
        selected_modules = normalize_modules(modules)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if preprocessing not in PREPROCESSING_MODES:
        raise HTTPException(status_code=422, detail="Unsupported preprocessing mode")
    if legal_traffic_side not in {"left", "right"}:
        raise HTTPException(status_code=422, detail="legal_traffic_side must be left or right")
    try:
        parking_zone = parse_zone(no_parking_zone)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    original_filename = Path(file.filename or "upload.jpg").name
    suffix = Path(original_filename).suffix.lower() or ".jpg"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Unsupported image type")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    input_name = f"upload_{uuid4().hex}{suffix}"
    input_path = UPLOAD_DIR / input_name
    input_path.write_bytes(content)

    try:
        started_at = time.perf_counter()
        result = await run_in_threadpool(
            run_analysis,
            str(input_path),
            conf,
            stopline_y_ratio,
            selected_modules,
            preprocessing,
            parking_zone,
            legal_traffic_side,
        )
        processing_time_ms = round((time.perf_counter() - started_at) * 1000, 1)
        result["summary"]["processing_time_ms"] = processing_time_ms
        analysis = create_analysis(
            original_filename=original_filename,
            stored_input_name=input_name,
            confidence=conf,
            stopline_y_ratio=stopline_y_ratio,
            result=result,
        )
    except Exception as exc:
        input_path.unlink(missing_ok=True)
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})

    return {
        "success": True,
        "analysis_id": analysis.id,
        "created_at": analysis.created_at.isoformat(),
        "original_filename": original_filename,
        "annotated_image_url": f"/static/outputs/{result['annotated_name']}",
        "processed_image_url": (
            f"/static/outputs/{result['processed_name']}" if result.get("processed_name") else None
        ),
        "selected_modules": result["selected_modules"],
        "module_results": result["module_results"],
        "summary": result["summary"],
        "meta": result["meta"],
    }
