# A sales forecasting model using Decision Tree Regressor
# (c) Flowdiary ML Course Project

# import required libraries
import  pandas as pd
import joblib
import streamlit as st

#convert our project to streamlit app

# To load the model and encoder later, use:
model = joblib.load('model/sales_model.pkl')
encoder = joblib.load('model/sales_encoder.pkl')

# Accept user input for prediction
month = st.selectbox("Select month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
product = st.text_input("Enter product type (e.g., electronics): ")
holiday = st.selectbox("Was there a holiday? (0 or 1): ", [0,1])
promotion = st.selectbox("Was there a promotion? (0 or 1): ", [0,1])

if st.button("Predict Sales"):
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
    result = model.predict(predictdata)

    # print the result
    st.success(f"Prediction: {result}")

# (c) Flowdiary ML Course Project
    