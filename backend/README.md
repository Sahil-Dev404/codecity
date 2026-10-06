# codecity-backend

FastAPI service for **CodeCity**: accepts a repo URL, runs the mining → parsing →
graph → GNN pipeline (via `codecity-ml`) as a background job, and serves the
results for the 3D city frontend.

## Install

```bash
pip install -e ".[dev]"
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
# or from the repo root:
make dev-api   # Unix/Git Bash
tasks dev-api  # Windows cmd
```

Docs available at `http://localhost:8000/docs` once running.

## Layout

| Module | Responsibility |
|---|---|
| `core/` | Settings, logging, security (URL validation, SSRF guard), rate limiting |
| `api/v1/` | Route handlers: analyze, jobs, city, files, what-if, health |
| `schemas/` | Pydantic request/response models |
| `services/` | Business logic: orchestration, layout, inference, caching |
| `workers/` | Background job definitions (arq) |
| `db/` | SQLAlchemy session, models, repository |
| `utils/` | SSE helpers and small shared utilities |

## Tests

```bash
pytest tests -q
```

## Environment

Copy `.env.example` from the repo root to `.env` and adjust `DATABASE_URL`,
`REDIS_URL`, and the safety limits before running.