"""
Unit tests for Course Recommendation project.

Tests the course_recommendation.py script.
"""

import pytest
import pandas as pd
import os
import sys

# Add projects directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'projects', '03_course_recommendation'))


def test_course_data_exists():
    """Test that the course data file exists."""
    data_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'projects', 
        '03_course_recommendation', 
        'data', 
        'course_data.csv'
    )
    assert os.path.exists(data_path), "Course data file not found"


def test_course_data_loads():
    """Test that the course data can be loaded."""
    data_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'projects', 
        '03_course_recommendation', 
        'data', 
        'course_data.csv'
    )
    df = pd.read_csv(data_path)
    assert not df.empty, "DataFrame is empty"
    assert 'recommended_course' in df.columns, "Target column not found"


# Add more tests as needed
