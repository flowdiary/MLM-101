# Fraud Detection Project

Detect fraudulent credit card transactions using machine learning classification.

## Overview

This project demonstrates how to build a fraud detection system using classification algorithms. The model identifies potentially fraudulent transactions based on transaction patterns and features.

## Features

- Data preprocessing for imbalanced datasets
- Feature engineering for transaction data
- Classification model training
- Model evaluation (Precision, Recall, F1-Score)
- Confusion matrix analysis

## Files

- `fraud_detection.py`: Training script with model evaluation
- `data/fraud_data.csv`: Credit card transaction dataset
- `models/`: Directory for saved models (generated after training)
- `requirements.txt`: Project dependencies

## Dataset

**File:** `data/fraud_data.csv`

Contains credit card transaction features and fraud labels.

## Installation

```bash
cd projects/02_fraud_detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python fraud_detection.py
```

This will:

1. Load the transaction dataset
2. Handle class imbalance
3. Train the classification model
4. Evaluate performance
5. Display confusion matrix

## Model Performance

Evaluation metrics:

- **Accuracy**: Overall correctness
- **Precision**: Fraction of correct fraud predictions
- **Recall**: Fraction of frauds detected
- **F1-Score**: Harmonic mean of precision and recall

## Technologies Used

- Python 3.8+
- pandas
- scikit-learn
- joblib

## Future Improvements

- [ ] Handle imbalanced data (SMOTE, class weights)
- [ ] Try ensemble methods (Random Forest, Gradient Boosting)
- [ ] Add feature importance analysis
- [ ] Build web interface for real-time detection

## License

Part of the MLM-101 course. See [LICENSE](../../LICENSE) for details.

## Contact

For questions or issues:

- Course: https://flowdiary.com.ng/course/MLM-101
- Email: support@flowdiary.com.ng
