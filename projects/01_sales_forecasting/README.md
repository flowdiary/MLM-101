# Sales Forecasting Project

Predict future sales using machine learning regression techniques.

## Overview

This project demonstrates how to build a sales forecasting model using Decision Tree Regressor. The model predicts sales based on temporal features, product categories, and holiday indicators.

## Features

- Data preprocessing and feature engineering
- One-hot encoding for categorical variables
- Decision Tree Regressor model
- Model evaluation (R², MSE)
- Interactive Streamlit web application

## Files

- `sales_forecasting.py`: Training script with model evaluation
- `sales_app.py`: Streamlit web application for predictions
- `data/sales_data.csv`: Historical sales dataset
- `models/`: Directory for saved models (generated after training)
- `requirements.txt`: Project dependencies

## Dataset

**File:** `data/sales_data.csv`

**Features:**

- `month`: Month of the year
- `product`: Product category (electronics, clothing, etc.)
- `holiday`: Binary indicator (0 or 1)

**Target:** `sales` (amount to predict)

## Installation

```bash
cd projects/01_sales_forecasting

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python sales_forecasting.py
```

This will:

1. Load the dataset
2. Preprocess features
3. Train the model
4. Evaluate performance (R², MSE)
5. Accept user input for predictions

### Running the Web App

```bash
streamlit run sales_app.py
```

Then open your browser to `http://localhost:8501`

## Model Performance

The model is evaluated using:

- **R² Score**: Coefficient of determination
- **MSE**: Mean Squared Error

## Example Prediction

```python
# Example input
month = "January"
product = "electronics"
holiday = 1  # Yes

# Model predicts sales amount
```

## Technologies Used

- Python 3.8+
- pandas
- scikit-learn
- joblib
- streamlit

## Future Improvements

- [ ] Try other regression algorithms (Random Forest, XGBoost)
- [ ] Add cross-validation
- [ ] Include more features (marketing spend, seasonality)
- [ ] Deploy to cloud (Streamlit Cloud, Heroku)

## License

Part of the MLM-101 course. See [LICENSE](../../LICENSE) for details.

## Contact

For questions or issues:

- Course: https://flowdiary.com.ng/course/MLM-101
- Email: support@flowdiary.com.ng
