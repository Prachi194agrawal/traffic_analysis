# Traffic Analysis AI

A container-ready traffic violation analysis application with a React frontend, FastAPI backend, PostgreSQL persistence, and YOLO/EasyOCR inference.

## What it does

- Detects vehicles, number plates, helmets, seatbelts, and traffic lights
- Applies stop-line and red-light violation rules
- Produces an annotated image and detection metadata
- Stores every analysis and its detection rows in a database
- Exposes analysis history through the API

## Architecture

```text
Browser -> Nginx/React -> FastAPI -> PostgreSQL
                           |
                           +-> YOLO models / EasyOCR
```

In Docker, Nginx serves the frontend and proxies `/api/*` to FastAPI. Model weights, uploads, generated output, and PostgreSQL data use named volumes.

## Start with Docker

Requirements: Docker Engine and Docker Compose v2.

```bash
cp .env.example .env
# Set a strong POSTGRES_PASSWORD in .env
docker compose up --build -d
```

Open `http://localhost:8080` when using the example configuration. Check status with:

```bash
docker compose ps
curl http://localhost:8080/api/health
```

The first analysis can take several minutes because model weights may be downloaded and EasyOCR initializes on demand. To use private Hugging Face repositories, set `HF_TOKEN` in `.env`.

Stop the application while retaining data:

```bash
docker compose down
```

To also delete database records, models, uploads, and output volumes:

```bash
docker compose down --volumes
```

## Local development

The backend defaults to a local SQLite database when `DATABASE_URL` is not set.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

The frontend runs at `http://localhost:5173` and uses `http://127.0.0.1:8000` by default.

For a standalone PostgreSQL instance, use a SQLAlchemy connection URL such as:

```bash
export DATABASE_URL='postgresql+psycopg://traffic_app:password@localhost:5432/traffic_analysis'
```

Tables are created automatically when the API starts.

## API

- `GET /health` — application and database readiness
- `POST /analyze` — upload and analyze an image
- `GET /analyses?limit=20` — recent persisted analyses
- `GET /analyses/{id}` — one analysis with detection rows
- `GET /docs` — interactive OpenAPI documentation

When running through Docker, prefix these routes with `/api`, for example `http://localhost:8080/api/docs`.

## Model weights

The deployment includes these canonical model files in `backend/models/`:

```text
license_plate_best.pt
seatbelt_best.pt
helmet_best.pt
redlight_best.pt
vehicle_best.pt
```

These weights are tracked because every file is below GitHub's per-file limit. Docker copies them into the image, and a new `model_data` volume is initialized from that image. To replace a model, use the same canonical filename and rebuild the backend. Missing supported models are still downloaded on first use.

## Tests

The lightweight backend tests do not load ML dependencies or model weights.

```bash
python -m pip install -r backend/requirements-ci.txt
cd backend && pytest -q
cd ../frontend && npm ci && npm run build
docker compose config --quiet
```

## GitHub Actions

`.github/workflows/ci-cd.yml` runs on pull requests and pushes to `main`:

1. Runs FastAPI/database integration tests.
2. Builds the React frontend.
3. Builds the backend and frontend containers.
4. Publishes both images to GitHub Container Registry on non-PR runs.

Published image names are:

```text
ghcr.io/<owner>/<repo>-backend:latest
ghcr.io/<owner>/<repo>-frontend:latest
```

No deployment secrets are committed. Configure environment values on the deployment host with `.env`.

## Production notes

- Replace the example database password before deployment.
- Put TLS in front of port 80 using a cloud load balancer or reverse proxy.
- Back up the PostgreSQL and output volumes.
- Pin and scan model artifacts if the application handles sensitive evidence.
- Restrict `CORS_ORIGINS` if the backend is exposed independently from Nginx.
