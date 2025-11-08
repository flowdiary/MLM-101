# 🤖 Machine Learning Mastery (MLM-101)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Level](https://img.shields.io/badge/Level-Beginner%20to%20Advanced-brightgreen)](https://flowdiary.ai/course/MLM-101)
[![Maintained](https://img.shields.io/badge/Maintained-Yes-green.svg)](https://github.com/flowdiary/MLM-101)

> **A comprehensive, hands-on machine learning course from fundamentals to production deployment.**  
> Master Python, NumPy, Pandas, Scikit-learn, Deep Learning, NLP, and model deployment with real-world projects.

--- 

## 📚 Table of Contents

- [About](#-about)
- [Learning Outcomes](#-learning-outcomes)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
  - [Quick Start](#quick-start)
  - [Using Virtual Environment](#using-virtual-environment)
  - [Using Conda](#using-conda)
- [Repository Structure](#-repository-structure)
- [Course Content](#-course-content)
- [Interactive Notebooks](#-interactive-notebooks)
- [Learning Paths](#-learning-paths)
  - [Beginner Path](#-beginner-path-4-6-weeks)
  - [Intermediate Path](#-intermediate-path-6-8-weeks)
  - [Advanced Path](#-advanced-path-6-8-weeks)
  - [Project-Based Path](#-project-based-path-12-weeks)
- [Quick Start Guide by Goal](#-quick-start-guide-by-goal)
- [How to Use This Repository](#-how-to-use-this-repository)
  - [Running Jupyter Notebooks](#1-running-jupyter-notebooks)
  - [Running Projects](#2-running-projects)
  - [Running Deployment Apps](#3-running-deployment-apps)
- [Datasets](#-datasets)
- [Projects](#-projects)
- [Example Commands](#-example-commands)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

---

## 🧠 About

**Machine Learning Mastery (MLM-101)** is a complete educational program designed to take learners from absolute beginners to proficient machine learning practitioners. This repository contains all course materials including:

- 📘 **85+ Lecture Materials** covering ML theory, Python, NumPy, Pandas, Scikit-learn, Deep Learning, NLP, and Deployment
- 💻 **Hands-on Notebooks** with code examples and interactive labs
- 🚀 **Real-World Projects** including Sales Forecasting, Fraud Detection, and Course Recommendation systems
- 🌐 **Deployment Guides** for Streamlit, FastAPI, Gradio, and Docker
- 📊 **Datasets** for practical exercises

**Course Website:** [https://flowdiary.ai/course/MLM-101](https://flowdiary.ai/course/MLM-101)

### ✨ What's New (November 2025)

- ✅ **12 New Foundation Notebooks** covering Python, NumPy, Pandas, Matplotlib, and Scikit-Learn
- ✅ **100% Coverage** for all foundation topics (Lectures 7-46)
- ✅ **Comprehensive Learning Paths** for beginners to advanced learners
- ✅ **35+ Hands-on Notebooks** with ML examples and practice exercises
- ✅ **Complete Project Suite** with deployment examples

---

## 🎯 Learning Outcomes

By completing this course, you will be able to:

✅ Understand fundamental ML concepts (supervised, unsupervised, deep learning)  
✅ Master Python programming for data science and ML  
✅ Manipulate and analyze data using NumPy and Pandas  
✅ Visualize data effectively with Matplotlib  
✅ Build, train, and evaluate ML models with Scikit-learn  
✅ Develop deep learning models (CNNs, FFNNs) with Keras/TensorFlow  
✅ Implement NLP solutions including sentiment analysis and NER  
✅ Build RAG (Retrieval-Augmented Generation) systems with LangChain  
✅ Deploy ML models to production using Streamlit, FastAPI, and Docker  
✅ Apply ML to real-world problems through guided projects

---

## 📋 Prerequisites

### Required Knowledge

- Basic programming concepts (variables, loops, functions)
- High school level mathematics (algebra, basic statistics)
- Familiarity with command-line interfaces (recommended)

### Software Requirements

- **Python 3.8 or higher** ([Download Python](https://www.python.org/downloads/))
- **Jupyter Notebook** or **JupyterLab** (or VS Code with Jupyter extension)
- **Git** ([Download Git](https://git-scm.com/downloads))
- **Text Editor/IDE**: VS Code, PyCharm, or Jupyter Notebook
- **8GB+ RAM** recommended for deep learning notebooks

### Optional

- **Docker** for deployment modules ([Download Docker](https://www.docker.com/products/docker-desktop))
- **Anaconda/Miniconda** for environment management ([Download Anaconda](https://www.anaconda.com/download))

---

## 🚀 Installation

### Quick Start

1. **Clone the repository:**

   ```bash
   git clone https://github.com/flowdiary/MLM-101.git
   cd MLM-101
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

4. **Open a notebook and start learning!**

---

### Using Virtual Environment

**Recommended for isolating project dependencies.**

#### On macOS/Linux:

```bash
# Navigate to project directory
cd MLM-101

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

#### On Windows:

```bash
# Navigate to project directory
cd MLM-101

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

**To deactivate:**

```bash
deactivate
```

---

### Using Conda

**Recommended for data science workflows.**

```bash
# Create conda environment
conda create -n mlm101 python=3.10 -y

# Activate environment
conda activate mlm101

# Install dependencies
pip install -r requirements.txt

# Or use conda for main packages
conda install numpy pandas matplotlib scikit-learn jupyter -y
pip install streamlit tensorflow langchain

# Launch Jupyter
jupyter notebook
```

**To deactivate:**

```bash
conda deactivate
```

---

## 📂 Repository Structure

```
MLM-101/
├── docs/                      # Course handouts, slides, and guides
│   ├── handouts/             # PDF lecture materials
│   └── guides/               # Installation & troubleshooting guides
│
├── notebooks/                 # Jupyter notebooks organized by topic
│   ├── 01_basics/            # ML fundamentals
│   ├── 02_deep_learning/     # Neural networks, CNNs
│   ├── 03_nlp/               # NLP and text processing
│   ├── 04_rag/               # RAG systems
│   └── 05_deployment/        # Model deployment
│
├── projects/                  # Real-world ML projects
│   ├── 01_sales_forecasting/
│   ├── 02_fraud_detection/
│   └── 03_course_recommendation/
│
├── scripts/                   # Python scripts (converted notebooks)
│   ├── deployment/
│   └── data/
│
├── apps/                      # Deployment applications
│   ├── fastapi_app/          # REST API
│   ├── gradio_app/           # RAG UI
│   └── streamlit_app/        # Interactive frontend
│
├── data/                      # Dataset storage (see data/README.md)
├── models/                    # Saved model storage
├── docker/                    # Docker configurations
├── tests/                     # Unit tests
│
├── requirements.txt           # Python dependencies
├── LICENSE                    # Apache 2.0 License
└── README.md                  # This file
```

---

## 📖 Course Content

### **Phase 1: Machine Learning Theory**

Introduction to ML, AI vs ML vs DL, Neural Networks, Algorithm Types, ML System Building

### **Phase 2: Python Programming for ML**

Variables, Data Types, Control Flow, Loops, Data Structures (Lists, Tuples, Sets, Dicts), Functions, OOP, Modules

### **Phase 3: NumPy for Data Computing**

Arrays, Mathematical Operations, Matrices, Linear Algebra, Random & Probability

### **Phase 4: Pandas for Data Analysis**

DataFrames, CSV/JSON I/O, Data Cleaning, Engineering, Analysis

### **Phase 5: Data Visualization with Matplotlib**

Plots, Customization, Sales Visualization, Exporting

### **Phase 6: Machine Learning with Scikit-Learn**

Datasets, Model Training, Preprocessing, Encoding, Scaling, Algorithms (Regression, Classification), Evaluation, Ensembles, Hyperparameter Tuning

### **Phase 7: Deep Learning**

Feedforward Neural Networks (FFNN), Backpropagation, Activation Functions

### **Phase 8: Natural Language Processing (NLP)**

Text Preprocessing, Sentiment Analysis, Named Entity Recognition (NER), Sequence Models

### **Phase 9: Convolutional Neural Networks (CNN)**

CNN Architecture, Padding, Pooling, Image Classification, Transfer Learning (ResNet50, VGG16)

### **Phase 10: RAG (Retrieval-Augmented Generation)**

RAG Systems, LangChain, Pinecone, ChromaDB

### **Phase 11: Deployment**

Model Serialization, FastAPI, Streamlit, Docker, Cloud Hosting

**Total: 85 Lectures**

---

## � Interactive Notebooks

### Complete Notebook Collection (35+ Notebooks)

All notebooks are production-ready with executable code, ML examples, and practice exercises.

#### **01_basics/** - Foundation & Preprocessing (14 notebooks)

**Python Programming Fundamentals:**

1. 📘 `python_basics.ipynb` - Variables, data types, operators, conditionals
2. 📘 `python_control_flow.ipynb` - Loops, iterations, list comprehensions
3. 📘 `python_data_structures.ipynb` - Lists, tuples, sets, dictionaries
4. 📘 `python_functions_oop.ipynb` - Functions, lambda, classes, inheritance

**NumPy for Numerical Computing:** 5. 🔢 `numpy_arrays_basics.ipynb` - Arrays, operations, indexing, broadcasting 6. 🔢 `numpy_linear_algebra.ipynb` - Matrix operations, eigenvalues, PCA

**Pandas for Data Manipulation:** 7. 📊 `pandas_dataframes_basics.ipynb` - DataFrames, Series, reading data 8. 📊 `pandas_data_cleaning.ipynb` - Missing values, duplicates, outliers 9. 📊 `pandas_data_analysis.ipynb` - GroupBy, pivot tables, merging

**Matplotlib for Visualization:** 10. 📈 `matplotlib_plotting_basics.ipynb` - Line, scatter, bar, histograms 11. 📈 `matplotlib_customization.ipynb` - Colors, labels, annotations, styles

**Scikit-Learn Preprocessing:** 12. 🔧 `sklearn_preprocessing.ipynb` - Scaling, encoding, pipelines, train/test split

**ML Evaluation & Tuning:** 13. 📏 `accuracy_metrics.ipynb` - Metrics, confusion matrix, ROC curves 14. ⚙️ `hyperparameter_tuning.ipynb` - Grid search, random search, cross-validation

#### **02_deep_learning/** - Neural Networks (5 notebooks)

15. 🧠 `ffnn_classification.ipynb` - Feedforward neural networks for classification
16. 🧠 `deep_learning_lectures.ipynb` - Deep learning fundamentals
17. 🖼️ `cnn_image_classification.ipynb` - CNN architecture and image classification
18. 🔄 `transfer_learning_resnet50.ipynb` - Transfer learning with ResNet50
19. 🔄 `transfer_learning_vgg16.ipynb` - Transfer learning with VGG16

#### **03_nlp/** - Natural Language Processing (6 notebooks)

20. 📝 `nlp_introduction.ipynb` - NLP fundamentals and concepts
21. 🔤 `nlp_preprocessing.ipynb` - Tokenization, stemming, lemmatization
22. 💬 `sentiment_analysis_scikit.ipynb` - Sentiment classification with Scikit-learn
23. 🏷️ `named_entity_recognition.ipynb` - NER with spaCy/NLTK
24. 🔁 `sequence_models_nlp.ipynb` - RNNs, LSTMs for text
25. 📚 `text_representation_techniques.ipynb` - Bag-of-words, TF-IDF, embeddings

#### **04_rag/** - Retrieval-Augmented Generation (2 notebooks)

26. 🤖 `rag_langchain_book_pdf.ipynb` - RAG with PDF documents
27. 🗄️ `rag_langchain_pinecone_chromadb.ipynb` - Vector databases integration

#### **05_deployment/** - Model Deployment (6 notebooks)

28. 💾 `01_model_serialization.ipynb` - Pickle, joblib, model saving
29. 🌐 `02_serving_fastapi.ipynb` - REST API with FastAPI
30. 🎨 `03_rag_langchain_gradio.ipynb` - RAG UI with Gradio
31. 🐳 `04_docker_and_containerization.ipynb` - Docker for ML apps
32. ⚡ `05_real_time_inference.ipynb` - Real-time predictions
33. 🎯 `06_hands_on_lab_deploy_sentiment_or_cnn.ipynb` - Deployment lab

---

## 🎓 Learning Paths

### 🌱 Beginner Path (4-6 weeks)

**Week 1-2: Python & NumPy Foundations**

```
python_basics.ipynb
→ python_control_flow.ipynb
→ python_data_structures.ipynb
→ python_functions_oop.ipynb
→ numpy_arrays_basics.ipynb
→ numpy_linear_algebra.ipynb
```

**Week 3: Data Manipulation**

```
pandas_dataframes_basics.ipynb
→ pandas_data_cleaning.ipynb
→ pandas_data_analysis.ipynb
```

**Week 4: Data Visualization**

```
matplotlib_plotting_basics.ipynb
→ matplotlib_customization.ipynb
```

**Week 5-6: First ML Project**

```
sklearn_preprocessing.ipynb
→ accuracy_metrics.ipynb
→ projects/01_sales_forecasting/
```

### 🚀 Intermediate Path (6-8 weeks)

**Prerequisites:** Complete Beginner Path

**Week 1-2: Advanced Scikit-Learn**

```
hyperparameter_tuning.ipynb
→ Build classification models
→ projects/02_fraud_detection/
```

**Week 3-4: Deep Learning Basics**

```
deep_learning_lectures.ipynb
→ ffnn_classification.ipynb
→ cnn_image_classification.ipynb
```

**Week 5-6: Transfer Learning**

```
transfer_learning_resnet50.ipynb
→ transfer_learning_vgg16.ipynb
→ Custom image classification project
```

**Week 7-8: NLP Fundamentals**

```
nlp_introduction.ipynb
→ nlp_preprocessing.ipynb
→ sentiment_analysis_scikit.ipynb
→ named_entity_recognition.ipynb
```

### 🔥 Advanced Path (6-8 weeks)

**Prerequisites:** Complete Intermediate Path

**Week 1-2: Advanced NLP**

```
text_representation_techniques.ipynb
→ sequence_models_nlp.ipynb
→ Build custom NLP pipeline
```

**Week 3-4: RAG Systems**

```
rag_langchain_book_pdf.ipynb
→ rag_langchain_pinecone_chromadb.ipynb
→ projects/03_course_recommendation/
```

**Week 5-6: Model Deployment**

```
01_model_serialization.ipynb
→ 02_serving_fastapi.ipynb
→ 03_rag_langchain_gradio.ipynb
→ 04_docker_and_containerization.ipynb
```

**Week 7-8: Production ML**

```
05_real_time_inference.ipynb
→ 06_hands_on_lab_deploy_sentiment_or_cnn.ipynb
→ Deploy your own ML app
```

### 📊 Project-Based Path (12 weeks)

Focus on completing all three major projects with supporting notebooks:

**Weeks 1-4: Sales Forecasting**

- Foundation notebooks (Python, NumPy, Pandas, Matplotlib)
- Scikit-learn preprocessing
- Complete `projects/01_sales_forecasting/`
- Deploy with Streamlit

**Weeks 5-8: Fraud Detection**

- Deep learning notebooks
- Imbalanced data handling
- Complete `projects/02_fraud_detection/`
- Create FastAPI endpoint

**Weeks 9-12: Course Recommendation**

- NLP notebooks
- RAG system setup
- Complete `projects/03_course_recommendation/`
- Full stack deployment with Docker

---

## 🎯 Quick Start Guide by Goal

### Goal: "I want to learn Python for ML"

**Start here:**

1. `python_basics.ipynb`
2. `python_control_flow.ipynb`
3. `python_data_structures.ipynb`
4. `python_functions_oop.ipynb`

### Goal: "I want to analyze data"

**Prerequisites:** Python basics  
**Start here:**

1. `numpy_arrays_basics.ipynb`
2. `pandas_dataframes_basics.ipynb`
3. `pandas_data_cleaning.ipynb`
4. `pandas_data_analysis.ipynb`
5. `matplotlib_plotting_basics.ipynb`

### Goal: "I want to build ML models"

**Prerequisites:** Python + Data analysis  
**Start here:**

1. `sklearn_preprocessing.ipynb`
2. `accuracy_metrics.ipynb`
3. `hyperparameter_tuning.ipynb`
4. `projects/01_sales_forecasting/`

### Goal: "I want to work with images"

**Prerequisites:** Python + ML basics  
**Start here:**

1. `deep_learning_lectures.ipynb`
2. `cnn_image_classification.ipynb`
3. `transfer_learning_resnet50.ipynb`
4. `transfer_learning_vgg16.ipynb`

### Goal: "I want to work with text/NLP"

**Prerequisites:** Python + ML basics  
**Start here:**

1. `nlp_introduction.ipynb`
2. `nlp_preprocessing.ipynb`
3. `sentiment_analysis_scikit.ipynb`
4. `named_entity_recognition.ipynb`
5. `text_representation_techniques.ipynb`

### Goal: "I want to deploy ML models"

**Prerequisites:** ML models built  
**Start here:**

1. `01_model_serialization.ipynb`
2. `02_serving_fastapi.ipynb`
3. `03_rag_langchain_gradio.ipynb`
4. `04_docker_and_containerization.ipynb`

---

## �🛠️ How to Use This Repository

### 1. Running Jupyter Notebooks

```bash
# Activate your environment (venv or conda)
source venv/bin/activate  # or: conda activate mlm101

# Navigate to notebooks directory
cd notebooks

# Launch Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

**Tip:** Notebooks are organized by topic. Start with `01_basics/` if you're new to ML.

### 📖 Notebook Navigation Tips

**Each notebook includes:**

- 📚 **Learning Objectives** - What you'll learn
- 💻 **Executable Code** - Run cells to see results
- 🎯 **ML Examples** - Real-world use cases
- ✏️ **Practice Exercises** - Test your knowledge
- 📝 **Solutions** - Complete exercise solutions
- 🔗 **Next Steps** - Suggested follow-up notebooks

**How to navigate:**

```python
# In Jupyter Notebook/Lab:
# - Shift + Enter: Run cell and move to next
# - Ctrl/Cmd + Enter: Run cell
# - B: Create new cell below
# - A: Create new cell above
# - M: Convert to Markdown
# - Y: Convert to Code
```

**Recommended workflow:**

1. Read the learning objectives
2. Run each code cell in order
3. Modify examples to experiment
4. Complete practice exercises
5. Check solutions
6. Move to the next notebook in the learning path

### 2. Running Projects

Each project has its own directory with a README, code, and data.

**Example: Sales Forecasting Project**

```bash
# Navigate to project
cd projects/01_sales_forecasting

# Install project-specific dependencies (if any)
pip install -r requirements.txt

# Run the training script
python sales_forecasting.py

# Or run the Streamlit app
streamlit run sales_app.py
```

### 3. Running Deployment Apps

**FastAPI Example:**

```bash
cd apps/fastapi_app
pip install -r requirements.txt
uvicorn app:app --reload
# Visit: http://127.0.0.1:8000/docs
```

**Streamlit Example:**

```bash
cd apps/streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

**Gradio Example:**

```bash
cd apps/gradio_app
pip install -r requirements.txt
python app.py
```

### 📋 Notebook Completion Checklist

Track your progress through the course:

**Foundation (14 notebooks) - Est. 20-30 hours:**

- [ ] Python Basics (4 notebooks)
- [ ] NumPy (2 notebooks)
- [ ] Pandas (3 notebooks)
- [ ] Matplotlib (2 notebooks)
- [ ] Scikit-Learn Preprocessing (1 notebook)
- [ ] ML Metrics & Tuning (2 notebooks)

**Deep Learning (5 notebooks) - Est. 10-15 hours:**

- [ ] FFNN & Deep Learning Fundamentals
- [ ] CNN Image Classification
- [ ] Transfer Learning (ResNet50, VGG16)

**NLP (6 notebooks) - Est. 12-18 hours:**

- [ ] NLP Introduction & Preprocessing
- [ ] Sentiment Analysis
- [ ] Named Entity Recognition
- [ ] Text Representation & Sequence Models

**Advanced Topics (8 notebooks) - Est. 12-15 hours:**

- [ ] RAG Systems (2 notebooks)
- [ ] Model Deployment (6 notebooks)

**Projects (3 projects) - Est. 20-30 hours:**

- [ ] Sales Forecasting
- [ ] Fraud Detection
- [ ] Course Recommendation

**Total Estimated Time: 70-110 hours** (self-paced)

---

## 📊 Datasets

All datasets are located in project-specific `data/` folders. Some datasets are included; others must be downloaded.

### Included Datasets:

- **Sales Data** (`projects/01_sales_forecasting/data/sales_data.csv`)
- **Fraud Data** (`projects/02_fraud_detection/data/fraud_data.csv`)
- **Course Data** (`projects/03_course_recommendation/data/course_data.csv`)

### External Datasets:

For large datasets (e.g., ImageNet, COCO), see `data/README.md` for download instructions.

**Example Download Script:**

```bash
cd data
python download_data.py
```

---

## 🔬 Projects

### 1. **Sales Forecasting** (`projects/01_sales_forecasting/`)

Predict future sales using Decision Tree Regressor.  
**Techniques:** Regression, OneHotEncoding, Model Evaluation (R², MSE)

### 2. **Fraud Detection** (`projects/02_fraud_detection/`)

Detect fraudulent credit card transactions.  
**Techniques:** Classification, Imbalanced Data Handling, Precision-Recall

### 3. **Course Recommendation** (`projects/03_course_recommendation/`)

Recommend courses based on user goals and hobbies.  
**Techniques:** Decision Trees, Categorical Encoding, Multi-class Classification

---

## 💡 Example Commands

```bash
# Clone repository
git clone https://github.com/flowdiary/MLM-101.git
cd MLM-101

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook

# Run a project script
cd projects/01_sales_forecasting
python sales_forecasting.py

# Run Streamlit app
cd projects/01_sales_forecasting
streamlit run sales_app.py

# Run FastAPI app
cd apps/fastapi_app
uvicorn app:app --reload

# Run tests
pytest tests/

# Docker deployment
cd docker
docker-compose up --build
```

---

## 🐛 Troubleshooting

### Common Issues

**1. `ModuleNotFoundError: No module named 'xyz'`**

```bash
# Ensure you're in the correct environment
source venv/bin/activate  # or: conda activate mlm101

# Install missing package
pip install xyz
```

**2. Jupyter Kernel Not Found**

```bash
# Install IPython kernel
python -m ipykernel install --user --name=mlm101
```

**3. Permission Denied (macOS/Linux)**

```bash
# Use pip with --user flag
pip install --user -r requirements.txt
```

**4. CUDA/GPU Issues (Deep Learning)**

```bash
# Verify TensorFlow GPU installation
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install CPU-only version if no GPU
pip install tensorflow-cpu
```

**5. Port Already in Use (Streamlit/FastAPI)**

```bash
# Change port for Streamlit
streamlit run app.py --server.port 8502

# Change port for FastAPI
uvicorn app:app --port 8001
```

### Additional Help

- Check `docs/guides/troubleshooting.md` for detailed solutions
- Open an issue: [GitHub Issues](https://github.com/flowdiary/MLM-101/issues)
- Contact instructors: [Flowdiary Support](https://flowdiary.ai)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

### Areas for Contribution:

- 🐛 Bug fixes
- 📝 Documentation improvements
- 🧪 New project examples
- 🌐 Translations
- 📊 Additional datasets

---

## 📜 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

You are free to:

- ✅ Use commercially
- ✅ Modify
- ✅ Distribute
- ✅ Use for private purposes

Under the conditions:

- ℹ️ Include license and copyright notice
- ℹ️ State changes made to the code

---

## 🙏 Acknowledgments

**Course Instructors:**

- **Muhammad Auwal Ahmad** - Co-founder, Flowdiary
- **Abdullahi Ahmad** - MLM Tutor, Flowdiary
  - 🌐 [Website](https://abdull6771.github.io/aahmad.github.io/)
  - 💼 [LinkedIn](https://www.linkedin.com/in/abdullahi-ahmad-babura-57653b205/)
  - 📧 [Email](mailto:abdulll8392@gmail.com)

**Contributors:**

- All students and community contributors

**Special Thanks:**

- Scikit-learn, TensorFlow, and PyTorch communities
- Open-source library maintainers

---

## 📧 Contact

- **Website:** [https://flowdiary.ai](https://flowdiary.ai)
- **Course Page:** [https://flowdiary.ai/course/MLM-101](https://flowdiary.ai/course/MLM-101)
- **GitHub:** [https://github.com/flowdiary/MLM-101](https://github.com/flowdiary/MLM-101)
- **Email:** hello@flowdiary.ai

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/flowdiary/MLM-101?style=social)](https://github.com/flowdiary/MLM-101/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/flowdiary/MLM-101?style=social)](https://github.com/flowdiary/MLM-101/network/members)

Made by [Flowdiary](https://flowdiary.ai)

</div>
