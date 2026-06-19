import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from database import (
    analysis_to_dict,
    create_analysis,
    database_is_ready,
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

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_pipeline():
    # ML libraries and model weights are intentionally loaded only on first analysis.
    from pipeline import TrafficPipeline

    return TrafficPipeline()


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


@app.get("/analyses")
def analysis_history(limit: int = 20):
    safe_limit = max(1, min(limit, 100))
    return {"items": [analysis_to_dict(item) for item in list_analyses(safe_limit)]}


@app.get("/analyses/{analysis_id}")
def analysis_detail(analysis_id: str):
    analysis = get_analysis(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis_to_dict(analysis, include_detections=True)


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    stopline_y_ratio: float = Form(0.72),
):
    if not 0.0 < conf <= 1.0:
        raise HTTPException(status_code=422, detail="conf must be between 0 and 1")
    if not 0.0 < stopline_y_ratio < 1.0:
        raise HTTPException(status_code=422, detail="stopline_y_ratio must be between 0 and 1")

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
        result = get_pipeline().analyze_image(
            image_path=str(input_path),
            conf=conf,
            stopline_y_ratio=stopline_y_ratio,
        )
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
        "annotated_image_url": f"/static/outputs/{result['annotated_name']}",
        "summary": result["summary"],
        "meta": result["meta"],
    }
