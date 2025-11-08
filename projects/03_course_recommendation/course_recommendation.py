# A Course Recommendation model using Decision Tree Classifier
# (c) Project ML Course Project

# Import required libraries
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("dataset/course_data.csv")

# Convert to DataFrame
df = pd.DataFrame(data)

print(df.head(10))

# Split into features and target variable
y = df['recommended_course']
X = df.drop(['recommended_course'], axis=1)

# Encode categorical columns (goal and hobby)
encoder = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['goal', 'hobby'])
    ],
    remainder='passthrough'
)

# Fit and transform the features
newX = encoder.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(newX, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model Accuracy: ", accuracy_score(y_test, y_pred))

# Accept user input for prediction
goal = input("Enter your goal (e.g., job, freelancing, business): ").lower()
hobby = input("Enter your hobby (e.g., Programming, Cryptocurrency, Design): ").capitalize()

# Create a DataFrame for prediction
predict_data = pd.DataFrame({
    'goal': [goal],
    'hobby': [hobby]
})

# Transform the input using the fitted encoder
predictX = encoder.transform(predict_data)

# Make a prediction
prediction = model.predict(predictX)
print("\n Recommended Course:", prediction[0])

# Save the model
#joblib.dump(model, 'model/course_model.pkl')

# Save the encoder
#joblib.dump(encoder, 'model/course_encoder.pkl')
