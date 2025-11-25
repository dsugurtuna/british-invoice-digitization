.PHONY: setup train run clean docker-build docker-run lint

PYTHON := python3
PIP := pip

setup:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt

train:
	@echo "Starting training pipeline..."
	cd notebooks && jupyter nbconvert --to python 01_train_yolov5_invoices.ipynb && $(PYTHON) 01_train_yolov5_invoices.py

# ============================================================================
# RoyalAudit Digitizer Makefile
# ============================================================================

.PHONY: help install test lint format clean docker-build docker-run dev

# Variables
PYTHON := python3
PIP := pip
DOCKER_IMAGE := royalaudit-digitizer
DOCKER_TAG := latest

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	$(PIP) install -e ".[dev,api]"
	$(PIP) install pre-commit
	pre-commit install

test:  ## Run tests
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:  ## Run linting
	ruff check .
	black --check .
	mypy src/

format:  ## Format code
	black .
	ruff check . --fix
	isort .

clean:  ## Clean up cache files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/

dev:  ## Run development server
	$(PYTHON) -m streamlit run src/app.py

docker-build:  ## Build Docker image
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:  ## Run Docker container
	docker run -p 8000:8000 $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-compose-up:  ## Run with Docker Compose
	docker-compose up --build -d

docker-compose-down:  ## Stop Docker Compose
	docker-compose down

lint:
	@echo "Running linter..."
	pylint src/*.py

docker-build:
	@echo "Building Docker image..."
	docker build -t royalaudit-digitizer:latest .

docker-run:
	@echo "Running Docker container..."
	docker run -p 8501:8501 royalaudit-digitizer:latest

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf runs/
