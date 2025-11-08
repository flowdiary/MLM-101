# Changelog

All notable changes to the Machine Learning Mastery (MLM-101) course repository will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Additional CNN architectures (MobileNet, EfficientNet)
- Time series forecasting project
- Advanced RAG patterns (multi-document, hybrid search)
- Kubernetes deployment guide
- MLOps best practices module

---

## [2.0.0] - 2025-01-XX (Proposed Restructure)

### Added

- **New README.md**: Comprehensive, student-friendly documentation with installation, usage, and troubleshooting
- **CONTRIBUTING.md**: Guidelines for contributors with code style, PR process, and development setup
- **.gitignore**: Proper exclusions for Python, Jupyter, OS files, and large binaries
- **Root requirements.txt**: Unified dependency management for all course modules
- **Structured folder layout**:
  - `docs/` for handouts and guides
  - `notebooks/` organized by topic (basics, deep learning, NLP, RAG, deployment)
  - `projects/` for student projects with individual READMEs
  - `apps/` for deployment applications
  - `scripts/` for utility scripts
  - `tests/` for unit tests
- **Dataset documentation**: `data/README.md` with download instructions
- **Model documentation**: `models/README.md` for saved models

### Changed

- **Reorganized notebooks**: Moved from scattered `resources/` to topic-based `notebooks/` structure
- **Renamed project scripts** for clarity:
  - `course_project.py` → `course_recommendation.py`
  - `sales_project.py` → `sales_forecasting.py`
  - `fraud_project.py` → `fraud_detection.py`
- **Moved deployment materials**: From `resources/deployment/` to root-level `apps/` and `scripts/`
- **Consolidated requirements**: Removed duplicate `requirements.txt` from individual project folders

### Removed

- Tracked `.DS_Store`, `__pycache__`, and `venv/` files from repository
- Large PDF files moved to Git LFS or external hosting recommendations
- Duplicate requirements files

### Fixed

- Missing setup instructions in README
- Inconsistent folder naming conventions
- Large binary files tracked in Git
- Missing `.gitignore` causing clutter

---

## [1.0.0] - 2024-XX-XX (Current State)

### Added

- Complete course curriculum (85 lectures)
- Phase 1-11: ML Theory, Python, NumPy, Pandas, Matplotlib, Scikit-learn, Deep Learning, NLP, CNN, RAG, Deployment
- Three student projects:
  - Sales Forecasting
  - Fraud Detection
  - Course Recommendation
- Deployment guides:
  - Streamlit applications
  - FastAPI REST APIs
  - Gradio interfaces
  - Docker containerization
- Jupyter notebooks for hands-on labs:
  - FFNN classification
  - CNN image classification
  - Transfer learning (ResNet50, VGG16)
  - NLP preprocessing and sentiment analysis
  - Named Entity Recognition
  - RAG with LangChain
- Course handouts and PDF materials
- Apache 2.0 License

### Course Content

- **Phase 1**: Machine Learning Theory (6 lectures)
- **Phase 2**: Python Programming (17 lectures)
- **Phase 3**: NumPy (8 lectures)
- **Phase 4**: Pandas (6 lectures)
- **Phase 5**: Matplotlib (5 lectures)
- **Phase 6**: Scikit-learn (16 lectures)
- **Phase 7**: Deep Learning (2 lectures)
- **Phase 8**: NLP (5 lectures)
- **Phase 9**: CNN (7 lectures)
- **Phase 10**: RAG (2 lectures)
- **Phase 11**: Deployment & Conclusion (6 lectures)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to this project.

## Maintainers

- Muhammad Auwal Ahmad - Co-founder, Flowdiary
- Abdullahi Ahmad Babura - MLM Tutor, Flowdiary

---

**Note**: Versions prior to 1.0.0 were internal development releases and are not documented here.
