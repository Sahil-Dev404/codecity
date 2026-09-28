# ─────────────────────────────────────────────────────────────
# CodeCity: developer commands.   Run `make help` to list them.
# ─────────────────────────────────────────────────────────────

VENV := .venv
ifeq ($(OS),Windows_NT)
	PY := $(VENV)/Scripts/python
else
	PY := $(VENV)/bin/python
endif

.DEFAULT_GOAL := help
.PHONY: help setup setup-py setup-web dev-api dev-web test test-py lint format train up down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────
setup: setup-py setup-web ## Install everything (Python + frontend)

setup-py: ## Create venv and install ml + backend packages (editable)
	python -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e "ml[dev]"
	$(PY) -m pip install -e "backend[dev]"

setup-web: ## Install frontend dependencies
	npm --prefix frontend install

# ── Run ──────────────────────────────────────────────────────
dev-api: ## Start FastAPI with auto-reload on :8000
	$(PY) -m uvicorn app.main:app --reload --port 8000 --app-dir backend

dev-web: ## Start Next.js dev server on :3000
	npm --prefix frontend run dev

# ── Quality ──────────────────────────────────────────────────
test: test-py ## Run all tests
	npm --prefix frontend run lint

test-py: ## Run Python tests (ml + backend)
	$(PY) -m pytest ml/tests backend/tests -q

lint: ## Static checks: ruff, black, mypy
	$(PY) -m ruff check ml backend
	$(PY) -m black --check ml backend
	$(PY) -m mypy ml/codecity_ml backend/app

format: ## Auto-format Python code
	$(PY) -m ruff check --fix ml backend
	$(PY) -m black ml backend

# ── ML ───────────────────────────────────────────────────────
train: ## Train GraphSAGE using ml/configs/train_sage.yaml
	$(PY) -m codecity_ml.cli train --config ml/configs/train_sage.yaml

# ── Docker ───────────────────────────────────────────────────
up: ## Start api + worker + redis + postgres + web
	docker compose up --build

down: ## Stop all containers
	docker compose down

# ── Housekeeping ─────────────────────────────────────────────
clean: ## Remove caches (keeps .venv and node_modules)
	find . -path ./$(VENV) -prune -o -path ./frontend/node_modules -prune -o -name __pycache__ -type d -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage