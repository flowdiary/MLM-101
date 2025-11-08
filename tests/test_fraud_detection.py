"""
Unit tests for Fraud Detection project.

Tests the fraud_detection.py script.
"""

import pytest
import pandas as pd
import os
import sys

# Add projects directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'projects', '02_fraud_detection'))


def test_fraud_data_exists():
    """Test that the fraud data file exists."""
    data_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'projects', 
        '02_fraud_detection', 
        'data', 
        'fraud_data.csv'
    )
    assert os.path.exists(data_path), "Fraud data file not found"


def test_fraud_data_loads():
    """Test that the fraud data can be loaded."""
    data_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'projects', 
        '02_fraud_detection', 
        'data', 
        'fraud_data.csv'
    )
    df = pd.read_csv(data_path)
    assert not df.empty, "DataFrame is empty"


# Add more tests as needed
