"""
Unit tests for Sales Forecasting project.

Tests the sales_forecasting.py script including:
- Data loading
- Preprocessing
- Model training
- Predictions
"""

import pytest
import pandas as pd
import os
import sys

# Add projects directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'projects', '01_sales_forecasting'))


def test_data_exists():
    """Test that the sales data file exists."""
    data_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'projects', 
        '01_sales_forecasting', 
        'data', 
        'sales_data.csv'
    )
    assert os.path.exists(data_path), "Sales data file not found"


def test_data_loads():
    """Test that the sales data can be loaded."""
    data_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'projects', 
        '01_sales_forecasting', 
        'data', 
        'sales_data.csv'
    )
    df = pd.read_csv(data_path)
    assert not df.empty, "DataFrame is empty"
    assert 'sales' in df.columns, "Target column 'sales' not found"


def test_data_columns():
    """Test that required columns exist in the dataset."""
    data_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'projects', 
        '01_sales_forecasting', 
        'data', 
        'sales_data.csv'
    )
    df = pd.read_csv(data_path)
    required_columns = ['month', 'product', 'holiday', 'sales']
    for col in required_columns:
        assert col in df.columns, f"Required column '{col}' not found"


def test_no_missing_values():
    """Test that there are no missing values in critical columns."""
    data_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'projects', 
        '01_sales_forecasting', 
        'data', 
        'sales_data.csv'
    )
    df = pd.read_csv(data_path)
    assert df['sales'].notna().all(), "Missing values found in sales column"


# Add more tests as needed:
# - test_model_training()
# - test_model_prediction()
# - test_model_accuracy()
