# Lecture 83 – Deployment: Deep Learning and RAG Models

This repository contains comprehensive materials for teaching model deployment in production environments.

## 📁 Structure

```
lecture_83_deployment/
├── notebooks/          # Jupyter notebooks with detailed explanations
├── scripts/           # Python scripts (notebook code converted to scripts)
├── apps/              # Production-ready applications
│   ├── fastapi_app/   # REST API for model serving
│   ├── gradio_app/    # RAG system with Gradio UI
│   └── streamlit_app/ # Interactive frontend
├── data/              # Sample data and download scripts
├── models/            # Saved models (created by running notebooks)
└── README.md         # This file
```

## 🎯 Learning Objectives

1. **Model Serialization**: Save and load models in production formats
2. **API Development**: Build REST APIs with FastAPI
3. **RAG Systems**: Implement retrieval-augmented generation
4. **Containerization**: Package applications with Docker
5. **Real-time Inference**: Handle streaming and batch predictions
6. **End-to-End Deployment**: Complete deployment pipeline

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip or conda
- Docker (optional, for containerization)
- 4GB+ RAM
- GPU (optional, but recommended)

### Installation

1. **Clone or download this repository**

2. **Create virtual environment** (recommended)

```bash
# Using conda
conda env create -f environment.yml
conda activate lecture83-deployment

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Verify installation**

```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
```

## 📚 Notebooks Overview

### 01 - Model Serialization

**File**: `notebooks/01_model_serialization.ipynb`  
**Runtime**: ~5 minutes  
**Topics**:

- Train a CNN on Fashion-MNIST
- Save models in HDF5 and SavedModel formats
- Serialize preprocessing artifacts
- Load and validate saved models
- Model versioning best practices

**Run**:

```bash
jupyter notebook notebooks/01_model_serialization.ipynb
# Or as script:
python scripts/01_model_serialization.py --output_dir models --subset_size 5000
```

### 02 - Serving with FastAPI

**File**: `notebooks/02_serving_fastapi.ipynb`  
**Runtime**: ~5 minutes  
**Topics**:

- Build production REST API
- Implement health and prediction endpoints
- Request validation with Pydantic
- API testing and benchmarking

**Run**:

```bash
# Start the server
cd apps/fastapi_app
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, run notebook
jupyter notebook notebooks/02_serving_fastapi.ipynb
```

### 03 - RAG with LangChain and Gradio

**File**: `notebooks/03_rag_langchain_gradio.ipynb`  
**Runtime**: ~7 minutes  
**Topics**:

- Build RAG pipeline with FAISS
- Implement semantic search
- Create Gradio UI for RAG system
- LLM integration patterns

**Run**:

```bash
jupyter notebook notebooks/03_rag_langchain_gradio.ipynb
# Or launch Gradio app:
python apps/gradio_app/app.py
```

### 04 - Docker and Containerization

**File**: `notebooks/04_docker_and_containerization.ipynb`  
**Runtime**: ~10 minutes  
**Topics**:

- Dockerfile creation
- Multi-service deployment with docker-compose
- Container networking and volumes
- Image optimization

**Run**:

```bash
jupyter notebook notebooks/04_docker_and_containerization.ipynb
# Or build directly:
cd apps/fastapi_app
docker build -t lecture83-fastapi .
docker run -p 8000:8000 lecture83-fastapi

# Push to Docker Hub (optional):
# 1. Login to Docker Hub
docker login

# 2. Tag the image with your Docker Hub username
docker tag lecture83-fastapi abdullbbr/lecture83-fastapi:latest

# 3. Push to Docker Hub
docker push abdullbbr/lecture83-fastapi:latest

# 4. Pull from Docker Hub (on any machine)
docker pull <your-username>/lecture83-fastapi:latest
docker run -p 8000:8000 <your-username>/lecture83-fastapi:latest
```

### 05 - Real-time Inference

**File**: `notebooks/05_real_time_inference.ipynb`  
**Runtime**: ~5 minutes  
**Topics**:

- Batch vs real-time inference
- WebSocket streaming
- Background tasks
- Performance monitoring

### 06 - Hands-on Lab

**File**: `notebooks/06_hands_on_lab_deploy_sentiment_or_cnn.ipynb`  
**Runtime**: ~15 minutes  
**Topics**:

- End-to-end deployment pipeline
- Model training to production
- API testing
- Deployment to cloud platforms

## 🛠️ Applications

### FastAPI App

**Location**: `apps/fastapi_app/`

```bash
cd apps/fastapi_app
pip install -r requirements.txt

# Development
uvicorn app:app --reload

# Production
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
```

**Endpoints**:

- `GET /ping` - Health check
- `POST /predict` - Make predictions
- `GET /metadata` - Model information
- `GET /docs` - Interactive API docs

### Gradio App

**Location**: `apps/gradio_app/`

```bash
cd apps/gradio_app
pip install -r requirements.txt
python app.py
```

Opens at: `http://localhost:7860`

### Streamlit App

**Location**: `apps/streamlit_app/`

```bash
cd apps/streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

Opens at: `http://localhost:8501`

## 🐳 Docker Deployment

### Build and Run FastAPI Container

```bash
cd apps/fastapi_app

# Build image
docker build -t lecture83-fastapi .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/../../models:/app/models \
  --name fastapi-server \
  lecture83-fastapi

# Check logs
docker logs -f fastapi-server

# Stop container
docker stop fastapi-server
```

### Docker Compose (Multi-Service)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🧪 Testing

### Run Unit Tests

```bash
# Install pytest
pip install pytest pytest-cov

# Run tests
pytest apps/fastapi_app/test_app.py -v

# With coverage
pytest apps/fastapi_app/test_app.py --cov=app --cov-report=html
```

### Manual API Testing

```bash
# Health check
curl http://localhost:8000/ping

# Get metadata
curl http://localhost:8000/metadata

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[0,0,...]]}'  # 28x28 image
```

## 📊 Project Makefile

Convenience commands for common tasks:

```bash
# Setup environment
make setup

# Train model
make train

# Start FastAPI server
make serve

# Build Docker image
make docker-build

# Run tests
make test

# Clean generated files
make clean
```

## 🔒 Security Notes

- **Never commit API keys** - Use environment variables
- **Validate all inputs** - FastAPI does this automatically with Pydantic
- **Rate limiting** - Implement in production
- **HTTPS** - Always use TLS in production
- **Model scanning** - Check for model poisoning

## 🌐 Deployment Platforms

### AWS

```bash
# ECR (Container Registry)
aws ecr create-repository --repository-name lecture83-fastapi
docker tag lecture83-fastapi:latest <account>.dkr.ecr.<region>.amazonaws.com/lecture83-fastapi
docker push <account>.dkr.ecr.<region>.amazonaws.com/lecture83-fastapi

# ECS/Fargate or EC2
# Follow AWS documentation for deploying containers
```

### Google Cloud

```bash
# GCR (Container Registry)
docker tag lecture83-fastapi gcr.io/<project-id>/lecture83-fastapi
docker push gcr.io/<project-id>/lecture83-fastapi

# Cloud Run
gcloud run deploy lecture83-fastapi \
  --image gcr.io/<project-id>/lecture83-fastapi \
  --platform managed
```

### Hugging Face Spaces (Gradio)

1. Create account on huggingface.co
2. Create new Space (select Gradio SDK)
3. Push your `apps/gradio_app/` code
4. Configure requirements.txt
5. App deploys automatically

## 📈 Monitoring

### Metrics to Track

- Request latency (p50, p95, p99)
- Error rate (4xx, 5xx)
- Model prediction distribution
- Resource usage (CPU, memory, GPU)
- Queue depth (for async systems)

### Tools

- **Prometheus** - Metrics collection
- **Grafana** - Dashboards
- **ELK Stack** - Log aggregation
- **Sentry** - Error tracking
- **DataDog** - All-in-one monitoring

## 🎓 Extension Ideas

1. **Model versioning** - A/B testing with multiple model versions
2. **Auto-scaling** - Kubernetes HPA based on request rate
3. **Model explainability** - Add SHAP or LIME endpoints
4. **Feedback loop** - Collect predictions for retraining
5. **Multi-modal** - Support text + image inputs
6. **Caching** - Redis for frequent predictions
7. **Async inference** - Queue-based system with Celery

## 📖 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [MLOps Principles](https://ml-ops.org/)
- [Production ML Course](https://madewithml.com/)

## 🐛 Troubleshooting

### Model not found

```bash
# Ensure you've run notebook 01 first to create models
python scripts/01_model_serialization.py
```

### Port already in use

```bash
# Find process using port
lsof -i :8000
# Kill process
kill -9 <PID>
```

### Docker permission issues

```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Re-login for changes to take effect
```

### GPU not detected

```bash
# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Install CUDA toolkit if needed
```

## 👥 Contributing

This is educational material. Feel free to:

- Report issues
- Suggest improvements
- Add more deployment examples
- Improve documentation

## 📄 License

This project is for educational purposes. Models and code are provided as-is.

## 🙏 Acknowledgments

- Fashion-MNIST dataset by Zalando Research
- TensorFlow and Keras teams
- FastAPI framework by Sebastián Ramírez
- Open-source ML community

---

**Lecture 83 - Deployment**  
_Teaching Deep Learning and RAG Model Deployment_

For questions or issues, please refer to the notebooks for detailed explanations.
