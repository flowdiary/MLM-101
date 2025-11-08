# Course Recommendation Project

Recommend courses to students based on their goals and interests using machine learning.

## Overview

This project demonstrates how to build a course recommendation system using Decision Tree Classifier. The model recommends appropriate courses based on student preferences, career goals, and hobbies.

## Features

- Multi-class classification
- Categorical feature encoding
- Decision Tree Classifier
- Model accuracy evaluation
- Interactive prediction interface

## Files

- `course_recommendation.py`: Training script with interactive predictions
- `data/course_data.csv`: Student preferences and course labels
- `models/`: Directory for saved models (generated after training)
- `requirements.txt`: Project dependencies

## Dataset

**File:** `data/course_data.csv`

**Features:**

- `goal`: Student's career goal (job, freelancing, business)
- `hobby`: Student's interest (Programming, Cryptocurrency, Design, etc.)

**Target:** `recommended_course` (course to recommend)

## Installation

```bash
cd projects/03_course_recommendation

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python course_recommendation.py
```

This will:

1. Load the student preference dataset
2. Encode categorical features
3. Train the Decision Tree Classifier
4. Evaluate model accuracy
5. Accept user input for course recommendations

## Example Usage

```
Enter your goal (e.g., job, freelancing, business): job
Enter your hobby (e.g., Programming, Cryptocurrency, Design): Programming

Recommended Course: Machine Learning Mastery
```

## Model Performance

The model is evaluated using:

- **Accuracy Score**: Percentage of correct recommendations

## Technologies Used

- Python 3.8+
- pandas
- scikit-learn
- joblib

## Course Categories

The model can recommend courses in various domains including:

- Machine Learning
- Data Science
- Web Development
- Mobile Development
- And more...

## Future Improvements

- [ ] Add more features (education level, experience)
- [ ] Try other algorithms (Random Forest, KNN)
- [ ] Build web interface with Streamlit
- [ ] Add course descriptions and links
- [ ] Implement collaborative filtering

## License

Part of the MLM-101 course. See [LICENSE](../../LICENSE) for details.

## Contact

For questions or issues:

- Course: https://flowdiary.com.ng/course/MLM-101
- Email: support@flowdiary.com.ng
