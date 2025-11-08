# 📁 MLM-101 Quick Reference - New Structure

## 🎯 Find What You Need

### 📓 I want to run a notebook

```bash
cd notebooks/

# Machine Learning Basics
notebooks/01_basics/accuracy_metrics.ipynb
notebooks/01_basics/hyperparameter_tuning.ipynb

# Deep Learning
notebooks/02_deep_learning/ffnn_classification.ipynb
notebooks/02_deep_learning/cnn_image_classification.ipynb
notebooks/02_deep_learning/transfer_learning_resnet50.ipynb

# Natural Language Processing
notebooks/03_nlp/nlp_introduction.ipynb
notebooks/03_nlp/sentiment_analysis_scikit.ipynb
notebooks/03_nlp/named_entity_recognition.ipynb

# RAG Systems
notebooks/04_rag/rag_langchain_book_pdf.ipynb
notebooks/04_rag/rag_langchain_pinecone_chromadb.ipynb

# Deployment
notebooks/05_deployment/01_model_serialization.ipynb
notebooks/05_deployment/02_serving_fastapi.ipynb
```

### 🚀 I want to run a project

```bash
# Sales Forecasting
cd projects/01_sales_forecasting
python sales_forecasting.py
# or
streamlit run sales_app.py

# Fraud Detection
cd projects/02_fraud_detection
python fraud_detection.py

# Course Recommendation
cd projects/03_course_recommendation
python course_recommendation.py
```

### 🌐 I want to run a deployment app

```bash
# FastAPI
cd apps/fastapi_app
uvicorn app:app --reload

# Streamlit
cd apps/streamlit_app
streamlit run app.py

# Gradio
cd apps/gradio_app
python app.py
```

### 📚 I need documentation

```bash
# Main README
less README.md

# Installation help
less docs/guides/installation.md

# Troubleshooting
less docs/guides/troubleshooting.md

# Dataset info
less docs/guides/dataset-guide.md

# Handouts
open docs/handouts/MLM-101-Handout.pdf
```

### 📊 I need a dataset

```bash
# Sales data
projects/01_sales_forecasting/data/sales_data.csv

# Fraud data
projects/02_fraud_detection/data/fraud_data.csv

# Course recommendation data
projects/03_course_recommendation/data/course_data.csv
```

### 🐳 I want to use Docker

```bash
cd docker/
docker-compose up
```

---

## 🗂️ Directory Purpose

| Directory    | What's Inside                                        |
| ------------ | ---------------------------------------------------- |
| `notebooks/` | All Jupyter notebooks, organized by topic            |
| `projects/`  | Student projects with data and code                  |
| `apps/`      | Deployment applications (FastAPI, Streamlit, Gradio) |
| `scripts/`   | Utility scripts for deployment and data              |
| `docs/`      | PDF handouts and documentation guides                |
| `data/`      | External datasets (with download instructions)       |
| `models/`    | Saved machine learning models                        |
| `tests/`     | Unit tests for projects                              |
| `docker/`    | Docker and containerization configs                  |

---

## 🔍 Quick Search

Looking for something specific?

```bash
# Find a file by name
find . -name "sentiment*"

# Search in all Python files
grep -r "DecisionTree" --include="*.py"

# Find all notebooks about NLP
ls notebooks/03_nlp/

# Find all READMEs
find . -name "README.md"
```

---

## ⚡ Common Commands

```bash
# Install everything
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook

# Run tests
pytest tests/

# Check Python version
python --version

# Activate virtual environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate      # Windows
```

---

## 🆘 Getting Help

1. **Check README**: `less README.md`
2. **Troubleshooting Guide**: `less docs/guides/troubleshooting.md`
3. **Project README**: Each project has its own README
4. **GitHub Issues**: Report problems
5. **Email**: support@flowdiary.com.ng

---

## 📍 Path Translation (Old → New)

| Old Path                               | New Path                                                     |
| -------------------------------------- | ------------------------------------------------------------ |
| `sales/sales_project.py`               | `projects/01_sales_forecasting/sales_forecasting.py`         |
| `fraud/fraud_project.py`               | `projects/02_fraud_detection/fraud_detection.py`             |
| `course/course_project.py`             | `projects/03_course_recommendation/course_recommendation.py` |
| `resources/accuracy_metrics.ipynb`     | `notebooks/01_basics/accuracy_metrics.ipynb`                 |
| `resources/deep-learning/ffnn_*.ipynb` | `notebooks/02_deep_learning/ffnn_*.ipynb`                    |
| `resources/deep-learning/nlp_*.ipynb`  | `notebooks/03_nlp/nlp_*.ipynb`                               |
| `resources/deployment/apps/`           | `apps/`                                                      |
| `resources/deployment/scripts/`        | `scripts/deployment/`                                        |

---

<div align="center">

**Save this file for quick reference!**

Print it: `cat QUICK_REFERENCE.md`

</div>
