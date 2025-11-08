.PHONY: help setup train serve docker-build docker-run test clean

help:
	@echo "Lecture 83 - Deployment Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup        - Create virtual environment and install dependencies"
	@echo "  make train        - Train the Fashion-MNIST model"
	@echo "  make serve        - Start the FastAPI server"
	@echo "  make serve-gradio - Start the Gradio RAG app"
	@echo "  make serve-streamlit - Start the Streamlit app"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make test         - Run unit tests"
	@echo "  make clean        - Remove generated files"
	@echo "  make notebooks    - Start Jupyter Lab"

setup:
	@echo "Setting up environment..."
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "✓ Setup complete! Activate with: source venv/bin/activate"

setup-conda:
	@echo "Setting up conda environment..."
	conda env create -f environment.yml
	@echo "✓ Setup complete! Activate with: conda activate lecture83-deployment"

train:
	@echo "Training Fashion-MNIST model..."
	python scripts/01_model_serialization.py --output_dir models --subset_size 5000 --epochs 2
	@echo "✓ Training complete! Models saved to models/"

serve:
	@echo "Starting FastAPI server..."
	cd apps/fastapi_app && uvicorn app:app --host 0.0.0.0 --port 8000 --reload

serve-gradio:
	@echo "Starting Gradio app..."
	cd apps/gradio_app && python app.py

serve-streamlit:
	@echo "Starting Streamlit app..."
	cd apps/streamlit_app && streamlit run app.py

docker-build:
	@echo "Building Docker image..."
	cd apps/fastapi_app && docker build -t lecture83-fastapi:latest .
	@echo "✓ Docker image built: lecture83-fastapi:latest"

docker-run:
	@echo "Running Docker container..."
	docker run -d \
		-p 8000:8000 \
		-v $(PWD)/models:/app/models \
		--name lecture83-server \
		lecture83-fastapi:latest
	@echo "✓ Container started! API available at http://localhost:8000"
	@echo "  View logs: docker logs -f lecture83-server"
	@echo "  Stop: docker stop lecture83-server"

docker-stop:
	@echo "Stopping Docker container..."
	docker stop lecture83-server || true
	docker rm lecture83-server || true

docker-compose-up:
	@echo "Starting all services with docker-compose..."
	docker-compose up -d
	@echo "✓ Services started!"

docker-compose-down:
	@echo "Stopping docker-compose services..."
	docker-compose down

test:
	@echo "Running unit tests..."
	pytest apps/fastapi_app/test_app.py -v --cov=apps/fastapi_app/app --cov-report=term-missing
	@echo "✓ Tests complete!"

test-api:
	@echo "Testing API endpoints..."
	@curl -s http://localhost:8000/ping | python -m json.tool
	@echo ""
	@curl -s http://localhost:8000/metadata | python -m json.tool

notebooks:
	@echo "Starting Jupyter Lab..."
	jupyter lab --notebook-dir=notebooks

format:
	@echo "Formatting code..."
	black scripts/ apps/
	@echo "✓ Code formatted!"

lint:
	@echo "Linting code..."
	flake8 scripts/ apps/ --max-line-length=100
	@echo "✓ Linting complete!"

clean:
	@echo "Cleaning generated files..."
	rm -rf models/*
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned!"

clean-docker:
	@echo "Cleaning Docker artifacts..."
	docker-compose down -v
	docker rmi lecture83-fastapi:latest || true
	@echo "✓ Docker cleaned!"

install-dev:
	@echo "Installing development dependencies..."
	pip install black flake8 mypy isort
	@echo "✓ Dev dependencies installed!"

check-gpu:
	@echo "Checking GPU availability..."
	@python -c "import tensorflow as tf; print(f'GPUs: {tf.config.list_physical_devices(\"GPU\")}')"

benchmark:
	@echo "Running API benchmark..."
	@python -c "import requests, time, numpy as np; \
		times = []; \
		for _ in range(50): \
			start = time.time(); \
			requests.post('http://localhost:8000/predict', json={'instances': [[[0]*28]*28]}); \
			times.append((time.time() - start) * 1000); \
		print(f'Mean latency: {np.mean(times):.2f}ms'); \
		print(f'P95 latency: {np.percentile(times, 95):.2f}ms')"

all: setup train serve
