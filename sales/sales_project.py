# A sales forecasting model using Decision Tree Regressor
# (c) Flowdiary ML Course Project

# import required libraries
import  pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error

# Load the dataset
data = pd.read_csv("dataset/sales_data.csv")

# Convert to df
df = pd.DataFrame(data)

print(df.head(10))

# Split the data into features and target variable
y = df['sales']
X = df.drop(['sales'], axis=1)

# Convert categorical columns
encoder = ColumnTransformer(
    transformers=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['month', 'product'])],
    remainder='passthrough'
)

# Fit and transform the features
newX = encoder.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(newX, y, test_size=0.2, random_state=42)

# Train the Decision Tree Regressor model
model = DecisionTreeRegressor()

# Fit the model
model.fit(X_train, y_train)

# Evaluate the model
print("R2: ", r2_score(y_test, model.predict(X_test)))
print("MSE: ", mean_squared_error(y_test, model.predict(X_test)))

# Accept user input for prediction
month = input("Enter month (e.g., January): ")
product = input("Enter product type (e.g., electronics): ")
holiday = int(input("Was there a holiday? (0 or 1): "))
promotion = int(input("Was there a promotion? (0 or 1): "))

# Make a prediction
predict_data = pd.DataFrame({
    "month": [month],
    "product": [product],
    "holiday": [holiday],
    "promotion": [promotion]
})

# Preprocess the prediction data
predict_data['month'] = predict_data['month'].str.lower()
predict_data['product'] = predict_data['product'].str.lower()

# Transform the prediction data using the fitted encoder
predictdata = encoder.transform(predict_data)

# Make the prediction
print("Prediction: ", model.predict(predictdata))

#save the model
joblib.dump(model, 'model/sales_model.pkl')

#save the encoder
joblib.dump(encoder, 'model/sales_encoder.pkl')

# (c) Flowdiary ML Course Project
