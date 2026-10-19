# ==============================================================================
# Smart Attendance AI Platform - Automation Makefile
# ==============================================================================

PYTHON = .venv/Scripts/python.exe
PIP = .venv/Scripts/pip.exe
UVICORN = .venv/Scripts/uvicorn.exe
STREAMLIT = .venv/Scripts/streamlit.exe
PYTEST = .venv/Scripts/pytest.exe

.PHONY: help setup run-backend run-frontend test seed evaluate clean

help:
	@echo "Available commands:"
	@echo "  make setup        - Install dependencies into virtual environment"
	@echo "  make seed         - Seed database with sample students and attendance logs"
	@echo "  make run-backend  - Start FastAPI backend server (port 8000)"
	@echo "  make run-frontend - Start Streamlit dashboard (port 8501)"
	@echo "  make test         - Run pytest test suite"
	@echo "  make evaluate     - Run benchmarking evaluation script"
	@echo "  make clean        - Remove Python cache and temporary artifacts"

setup:
	$(PIP) install -r requirements.txt

seed:
	$(PYTHON) -m scripts.seed_db

run-backend:
	$(UVICORN) backend.main:app --host 0.0.0.0 --port 8000 --reload

run-frontend:
	$(STREAMLIT) run frontend/app.py

test:
	$(PYTEST) -v

evaluate:
	$(PYTHON) scripts/evaluate.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
