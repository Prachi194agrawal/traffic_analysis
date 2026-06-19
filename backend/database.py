import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, URL, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, selectinload, sessionmaker
from sqlalchemy.pool import StaticPool


def _database_url() -> str | URL:
    if value := os.getenv("DATABASE_URL"):
        return value
    if host := os.getenv("POSTGRES_HOST"):
        return URL.create(
            "postgresql+psycopg",
            username=os.getenv("POSTGRES_USER", "traffic_app"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            host=host,
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "traffic_analysis"),
        )
    return "sqlite:///./traffic_analysis.db"


DATABASE_URL = _database_url()


class Base(DeclarativeBase):
    pass


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_input_name: Mapped[str] = mapped_column(String(255))
    annotated_name: Mapped[str] = mapped_column(String(255))
    confidence: Mapped[float] = mapped_column(Float)
    stopline_y_ratio: Mapped[float] = mapped_column(Float)
    final_status: Mapped[str] = mapped_column(String(80), index=True)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )

    detections: Mapped[list["Detection"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    module: Mapped[str] = mapped_column(String(80))
    class_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    ocr_text: Mapped[str | None] = mapped_column(String(255), nullable=True)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    rule: Mapped[str | None] = mapped_column(String(120), nullable=True)
    status: Mapped[str | None] = mapped_column(String(160), nullable=True)

    analysis: Mapped[Analysis] = relationship(back_populates="detections")


engine_options: dict[str, Any] = {"pool_pre_ping": True}
if str(DATABASE_URL).startswith("sqlite"):
    engine_options["connect_args"] = {"check_same_thread": False}
    if ":memory:" in DATABASE_URL:
        engine_options["poolclass"] = StaticPool

engine = create_engine(DATABASE_URL, **engine_options)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_database() -> None:
    Base.metadata.create_all(bind=engine)


def database_is_ready() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def create_analysis(
    *,
    original_filename: str,
    stored_input_name: str,
    confidence: float,
    stopline_y_ratio: float,
    result: dict[str, Any],
) -> Analysis:
    summary = dict(result["summary"])
    analysis = Analysis(
        original_filename=original_filename,
        stored_input_name=stored_input_name,
        annotated_name=result["annotated_name"],
        confidence=confidence,
        stopline_y_ratio=stopline_y_ratio,
        final_status=str(summary.get("final_status", "unknown")),
        summary=summary,
    )

    for row in result["meta"]:
        analysis.detections.append(
            Detection(
                module=str(row.get("module", "unknown")),
                class_name=row.get("class_name"),
                confidence=row.get("confidence"),
                bbox=row.get("bbox"),
                ocr_text=row.get("ocr_text"),
                ocr_confidence=row.get("ocr_confidence"),
                rule=row.get("rule"),
                status=row.get("status"),
            )
        )

    with SessionLocal() as session:
        session.add(analysis)
        session.commit()
        session.refresh(analysis)
    return analysis


def list_analyses(limit: int = 20) -> list[Analysis]:
    with SessionLocal() as session:
        statement = select(Analysis).order_by(Analysis.created_at.desc()).limit(limit)
        return list(session.scalars(statement))


def get_analysis(analysis_id: str) -> Analysis | None:
    with SessionLocal() as session:
        statement = (
            select(Analysis)
            .options(selectinload(Analysis.detections))
            .where(Analysis.id == analysis_id)
        )
        return session.scalar(statement)


def analysis_to_dict(analysis: Analysis, include_detections: bool = False) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": analysis.id,
        "original_filename": analysis.original_filename,
        "annotated_image_url": f"/static/outputs/{analysis.annotated_name}",
        "confidence": analysis.confidence,
        "stopline_y_ratio": analysis.stopline_y_ratio,
        "final_status": analysis.final_status,
        "summary": analysis.summary,
        "created_at": analysis.created_at.isoformat(),
    }
    if include_detections:
        data["meta"] = [
            {
                "module": item.module,
                "class_name": item.class_name,
                "confidence": item.confidence,
                "bbox": item.bbox,
                "ocr_text": item.ocr_text,
                "ocr_confidence": item.ocr_confidence,
                "rule": item.rule,
                "status": item.status,
            }
            for item in analysis.detections
        ]
    return data
