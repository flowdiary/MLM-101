# Lecture 83 - Deployment: Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Prerequisites

- Python 3.9+
- 4GB+ RAM
- Terminal/Command Prompt

### Step 1: Install Dependencies

```bash
# Navigate to project
cd lecture_83_deployment

# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt
```

### Step 2: Train a Model

```bash
# Option A: Run notebook
jupyter notebook notebooks/01_model_serialization.ipynb

# Option B: Run script
python scripts/01_model_serialization.py --subset_size 5000
```

This creates model files in `models/` directory (~5 minutes)

### Step 3: Start API Server

```bash
cd apps/fastapi_app
uvicorn app:app --reload
```

Server starts at: http://localhost:8000

### Step 4: Test the API

```bash
# In a new terminal
curl http://localhost:8000/ping

# View interactive docs
open http://localhost:8000/docs
```

### Step 5: Run Gradio RAG App

```bash
cd apps/gradio_app
python app.py
```

Opens at: http://localhost:7860

### Step 6: Run Streamlit App

```bash
cd apps/streamlit_app
streamlit run app.py
```

Opens at: http://localhost:8501

---

## 📚 Learning Path

### For Beginners

1. Start with `01_model_serialization.ipynb`
2. Progress through notebooks in order (01 → 06)
3. Complete the hands-on lab (notebook 06)

### For Intermediate Users

- Focus on notebooks 02 (FastAPI) and 03 (RAG)
- Build your own model and deploy it
- Experiment with Docker (notebook 04)

### For Advanced Users

- Implement all extension ideas from notebooks
- Deploy to cloud (AWS/GCP/Azure)
- Set up CI/CD pipeline
- Add monitoring and logging

---

## 🛠️ Makefile Shortcuts

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

---

## 📖 Notebook Overview

| Notebook | Topic               | Time   | Difficulty  |
| -------- | ------------------- | ------ | ----------- |
| 01       | Model Serialization | 5 min  | ⭐ Easy     |
| 02       | FastAPI Serving     | 5 min  | ⭐⭐ Medium |
| 03       | RAG + Gradio        | 7 min  | ⭐⭐ Medium |
| 04       | Docker              | 10 min | ⭐⭐⭐ Hard |
| 05       | Real-Time Inference | 5 min  | ⭐⭐ Medium |
| 06       | Hands-On Lab        | 15 min | ⭐⭐⭐ Hard |

---

## 🐛 Common Issues

### Model Not Found

```bash
# Run model training first
python scripts/01_model_serialization.py
```

### Port Already in Use

```bash
# Find process
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill process or use different port
uvicorn app:app --port 8001
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Docker Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

---

## 🎯 Learning Objectives Checklist

After completing this lecture, you should be able to:

- [ ] Serialize models in multiple formats
- [ ] Build REST APIs with FastAPI
- [ ] Implement RAG systems
- [ ] Containerize applications with Docker
- [ ] Deploy models to production
- [ ] Monitor and benchmark APIs
- [ ] Write deployment documentation

---

## 📊 Project Structure

```
lecture_83_deployment/
├── notebooks/          # 6 Jupyter notebooks
├── scripts/           # 6 Python scripts
├── apps/              # 3 web applications
│   ├── fastapi_app/   # REST API
│   ├── gradio_app/    # RAG demo
│   └── streamlit_app/ # Interactive UI
├── models/            # Saved models (generated)
├── data/              # Sample data
├── Makefile           # Automation commands
├── requirements.txt   # Dependencies
└── README.md         # Full documentation
```

---

## 🌐 Deployment Platforms

### Local

✓ Already working!

### Docker

```bash
make docker-build
make docker-run
```

### Hugging Face Spaces (Gradio)

1. Create account at huggingface.co
2. Create new Space (Gradio)
3. Upload `apps/gradio_app/` files
4. Auto-deploys!

### AWS EC2

```bash
# SSH to instance
ssh -i key.pem ubuntu@<IP>

# Install Docker
sudo apt-get update
sudo apt-get install docker.io

# Run container
docker run -p 8000:8000 your-image
```

### Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT/app

# Deploy
gcloud run deploy --image gcr.io/PROJECT/app
```

---

## 💡 Next Steps

1. **Complete all notebooks** (2-3 hours)
2. **Customize the model** (use your own dataset)
3. **Add features** (authentication, caching, etc.)
4. **Deploy to cloud** (AWS, GCP, or Azure)
5. **Set up monitoring** (Prometheus + Grafana)
6. **Build CI/CD** (GitHub Actions)

---

## 🙋 Getting Help

- Review the README.md for detailed docs
- Check notebook markdown cells for explanations
- Look at inline code comments
- Refer to troubleshooting section

---

## 🎓 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TensorFlow Serving Guide](https://www.tensorflow.org/tfx/guide/serving)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [MLOps Principles](https://ml-ops.org/)

---

**Happy Learning! 🚀**

Questions? Review the notebooks or check the README.md for more details.
