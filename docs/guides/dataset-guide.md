# Dataset Guide

## Included Datasets

All project datasets are included in the repository under their respective project directories:

### 1. Sales Forecasting Data

**Location:** `projects/01_sales_forecasting/data/sales_data.csv`

**Description:** Historical sales data with features including month, product type, and holiday indicators.

**Columns:**

- `month`: Month of sale
- `product`: Product category
- `holiday`: Holiday indicator (0 or 1)
- `sales`: Sales amount (target variable)

**Usage:**

```python
import pandas as pd
df = pd.read_csv('projects/01_sales_forecasting/data/sales_data.csv')
```

### 2. Fraud Detection Data

**Location:** `projects/02_fraud_detection/data/fraud_data.csv`

**Description:** Credit card transaction data for fraud detection.

**Usage:**

```python
import pandas as pd
df = pd.read_csv('projects/02_fraud_detection/data/fraud_data.csv')
```

### 3. Course Recommendation Data

**Location:** `projects/03_course_recommendation/data/course_data.csv`

**Description:** Student preferences and course recommendations.

**Columns:**

- `goal`: Student's career goal
- `hobby`: Student's hobby/interest
- `recommended_course`: Recommended course (target variable)

**Usage:**

```python
import pandas as pd
df = pd.read_csv('projects/03_course_recommendation/data/course_data.csv')
```

## External Datasets

For deep learning exercises requiring large datasets, download instructions are provided in the respective notebooks:

### Image Classification

- **ImageNet subset**: Instructions in `notebooks/02_deep_learning/cnn_image_classification.ipynb`
- **CIFAR-10**: Automatically downloaded by TensorFlow/Keras

### NLP Datasets

- **IMDB Reviews**: Automatically downloaded by TensorFlow/Keras
- **Custom PDFs for RAG**: See `notebooks/04_rag/` notebooks

### Download Script

For datasets not included in the repository:

```bash
cd scripts/data
python download_data.py
```

## Data Format Standards

All CSV files follow these conventions:

- **Encoding:** UTF-8
- **Separator:** Comma (`,`)
- **Header:** First row contains column names
- **Missing Values:** Indicated by empty cells or `NaN`

## Data Storage Best Practices

- ✅ Keep datasets under 10MB in Git
- ✅ Use `.gitignore` for large datasets
- ✅ Document download instructions for external data
- ✅ Store processed data in project-specific `data/` folders
- ✅ Never commit sensitive or private data

## Need Help?

- Check notebook documentation for dataset-specific information
- Contact: support@flowdiary.com.ng
- See: [README.md](../../README.md#-datasets)
