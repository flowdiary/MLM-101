# A Credit Card Fraud Detection model using Random Forest Classifier
# (c) Flowdiary ML Course Project

# Import required libraries
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("dataset/fraud_data.csv")

# Convert to DataFrame
df = pd.DataFrame(data)

print(df.head(10))

# Drop unwanted columns
df = df.drop(['transaction_id', 'timestamp', 'merchant_id', 'customer_id'], axis=1)

# Split into features and target
y = df['fraud_label']
X = df.drop(['fraud_label'], axis=1)

# Encode categorical columns
encoder = ColumnTransformer(
    transformers=[(
        'onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
        ['transaction_type', 'device_type', 'location']
    )],
    remainder='passthrough'
)

# Fit and transform the features
newX = encoder.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(newX, y, test_size=0.3, random_state=42)

# Train the Random Forest Classifier model
model = RandomForestClassifier()

# Fit the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))

# Accept user input for prediction
amount = float(input("Enter transaction amount: "))
transaction_type = input("Enter transaction type (e.g., withdrawal, purchase, transfer): ")
device_type = input("Enter device type (web or mobile): ")
location = input("Enter location (e.g., US, UK, CA, AU, IN): ")
previous_fraud = int(input("Previous fraud? (0 = No, 1 = Yes): "))

# Create prediction dataframe
predict_data = pd.DataFrame({
    'amount': [amount],
    'transaction_type': [transaction_type.lower()],
    'device_type': [device_type.lower()],
    'location': [location.upper()],
    'previous_fraud': [previous_fraud]
})

# Transform the prediction data using the fitted encoder
predictX = encoder.transform(predict_data)

# Make prediction
prediction = model.predict(predictX)

if prediction[0] == 1:
    print("FRAUD DETECTED")
else:
    print("Legitimate Transaction")

# Save the model
joblib.dump(model, 'model/fraud_model.pkl')

# Save the encoder
joblib.dump(encoder, 'model/fraud_encoder.pkl')

# (c) Flowdiary ML Course Project
