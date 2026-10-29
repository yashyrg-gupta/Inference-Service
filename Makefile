.PHONY: help setup download run-backend run-frontend docker-up docker-down docker-logs test

PYTHON=python3
PIP=pip3
BACKEND_DIR=backend

help:
	@echo "Diabetes Inference Service CLI"
	@echo ""
	@echo "  make setup          Install Python & Node dependencies"
	@echo "  make download       Download models from HuggingFace to models/"
	@echo "  make run-backend    Start FastAPI backend (hot reload)"
	@echo "  make run-frontend   Start React dev server"
	@echo "  make docker-up      Build and start Docker stack (DB, Backend, Frontend)"
	@echo "  make docker-down    Stop all Docker containers"
	@echo "  make docker-logs    Tail container logs"
	@echo "  make test           Run pytest test suite"

setup:
	cd $(BACKEND_DIR) && $(PIP) install -r requirements.txt
	cd frontend && npm install

download:
	cd $(BACKEND_DIR) && $(PYTHON) scripts/download_model.py

run-backend:
	cd $(BACKEND_DIR) && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	cd frontend && npm start

docker-up:
	docker compose up --build -d
	@echo "Services started:"
	@echo "  Backend API: http://localhost:8000/docs"
	@echo "  Frontend:    http://localhost:3000"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

test:
	docker compose exec -T backend pytest tests -v --tb=short || (cd $(BACKEND_DIR) && pytest -v --tb=short)
