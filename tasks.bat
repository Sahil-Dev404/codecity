@echo off
REM ─────────────────────────────────────────────────────────────
REM CodeCity: Windows equivalent of the Makefile.
REM Usage:  tasks <command>      e.g.  tasks setup
REM ─────────────────────────────────────────────────────────────
setlocal
set PY=.venv\Scripts\python.exe

if "%1"=="" goto help
goto %1

:help
echo.
echo   CodeCity commands:
echo     tasks setup      Install everything (Python + frontend)
echo     tasks setup-py   Create venv, install ml + backend (editable)
echo     tasks setup-web  Install frontend dependencies
echo     tasks dev-api    Start FastAPI with reload on :8000
echo     tasks dev-web    Start Next.js dev server on :3000
echo     tasks test       Run all tests
echo     tasks lint       ruff + black + mypy
echo     tasks format     Auto-format Python code
echo     tasks train      Train GraphSAGE
echo     tasks up         docker compose up --build
echo     tasks down       docker compose down
echo     tasks clean      Remove caches
echo.
goto :eof

:setup
call :setup-py
call :setup-web
goto :eof

:setup-py
if not exist .venv python -m venv .venv
%PY% -m pip install --upgrade pip
%PY% -m pip install -e "ml[dev]"
%PY% -m pip install -e "backend[dev]"
goto :eof

:setup-web
npm --prefix frontend install
goto :eof

:dev-api
%PY% -m uvicorn app.main:app --reload --port 8000 --app-dir backend
goto :eof

:dev-web
npm --prefix frontend run dev
goto :eof

:test
%PY% -m pytest ml/tests backend/tests -q
npm --prefix frontend run lint
goto :eof

:lint
%PY% -m ruff check ml backend
%PY% -m black --check ml backend
%PY% -m mypy ml/codecity_ml backend/app
goto :eof

:format
%PY% -m ruff check --fix ml backend
%PY% -m black ml backend
goto :eof

:train
%PY% -m codecity_ml.cli train --config ml/configs/train_sage.yaml
goto :eof

:up
docker compose up --build
goto :eof

:down
docker compose down
goto :eof

:clean
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
if exist .pytest_cache rd /s /q .pytest_cache
if exist .ruff_cache rd /s /q .ruff_cache
if exist .mypy_cache rd /s /q .mypy_cache
if exist htmlcov rd /s /q htmlcov
echo Caches removed.
goto :eof