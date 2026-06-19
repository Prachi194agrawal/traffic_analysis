# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements-base.txt backend/requirements.txt ./

# Keep the expensive ML downloads in separate cached layers. Ultralytics is
# installed without dependencies because requirements-base.txt supplies them
# with headless OpenCV, avoiding a second 70+ MB GUI OpenCV download.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip \
    for attempt in 1 2 3 4 5; do \
        pip install -r requirements-base.txt && break; \
        if [ "$attempt" = 5 ]; then exit 1; fi; \
        echo "Dependency download interrupted; retrying ($attempt/5)..."; \
        sleep 5; \
    done
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps "ultralytics>=8.3,<9.0"

COPY backend/ ./

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p uploads static/outputs models \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
