# Common development tasks. Each target is a thin wrapper around one command,
# so the commands in the README work the same with or without make.

PYTHON ?= python3
IMAGE ?= british-invoice-digitization:latest

.PHONY: help install install-model lint format typecheck test run docker-build clean

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-15s %s\n", $$1, $$2}'

install: ## Install the package with dev tools (no PyTorch needed)
	$(PYTHON) -m pip install -e ".[dev]"

install-model: ## Add the YOLOv5 runtime (CPU PyTorch wheels)
	$(PYTHON) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -e ".[yolov5]"

lint: ## Lint and check formatting
	ruff check .
	ruff format --check .

format: ## Apply formatting and safe lint fixes
	ruff format .
	ruff check --fix .

typecheck: ## Static type check
	mypy

test: ## Run the test suite with coverage
	pytest --cov --cov-fail-under=80

run: ## Start the API on localhost:8000 (needs trained weights, see README)
	uvicorn invoice_digitizer.api.main:app --host 127.0.0.1 --port 8000

docker-build: ## Build the container image
	docker build -t $(IMAGE) .

clean: ## Remove caches and build output
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
