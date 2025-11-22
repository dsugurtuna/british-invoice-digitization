.PHONY: setup train run clean docker-build docker-run lint

PYTHON := python3
PIP := pip

setup:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt

train:
	@echo "Starting training pipeline..."
	cd notebooks && jupyter nbconvert --to python 01_train_yolov5_invoices.ipynb && $(PYTHON) 01_train_yolov5_invoices.py

run:
	@echo "Launching Dashboard..."
	$(PYTHON) -m streamlit run src/app.py

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
