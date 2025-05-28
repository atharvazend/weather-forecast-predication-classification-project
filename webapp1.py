import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model, scaler, and label encoder
with open(r"C:\My folder\weather forecast predication classification project\logistic_regression_model.pkl", 'rb') as model_file:
    logistic_model = pickle.load(model_file)
with open(r"C:\My folder\weather forecast predication classification project\scaler.pkl", 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open(r"C:\My folder\weather forecast predication classification project\label_encoder.pkl", 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)


# Set up the Streamlit app title and description
st.set_page_config(page_title="Rain Prediction App", page_icon="🌧️")
st.title("🌧️ Weather Rain Prediction")
st.write("Enter the weather parameters below to predict if there will be rain or no rain.")

# Create input fields for the user
st.sidebar.header("Input Weather Parameters")

temp = st.sidebar.slider("Temperature (Celsius)", min_value=-10.0, max_value=40.0, value=25.0, step=0.1)
relative_humidity = st.sidebar.slider("Relative Humidity (percentage)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
wind_speed = st.sidebar.slider("Wind Speed (km/hr)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
cloud_cover = st.sidebar.slider("Cloud Cover (percentage)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
pressure = st.sidebar.slider("Pressure (milibars)", min_value=950.0, max_value=1100.0, value=1010.0, step=0.1)

# Prepare the input data for prediction
input_data = np.array([[temp, relative_humidity, wind_speed, cloud_cover, pressure]])

# Scale the input data using the loaded scaler
scaled_input_data = scaler.transform(input_data)

# Make prediction when the user clicks the button
if st.sidebar.button("Predict Weather"):
    prediction = logistic_model.predict(scaled_input_data)
    predicted_rain_status = encoder.inverse_transform(prediction)

    st.subheader("Prediction Result:")
    if predicted_rain_status[0] == 'rain':
        st.success(f"Based on the entered parameters, it is predicted to: **{predicted_rain_status[0].upper()}** ☔")
    else:
        st.info(f"Based on the entered parameters, it is predicted to: **{predicted_rain_status[0].upper()}** ☀️")

    st.write("---")
    st.subheader("Input Values:")
    input_df = pd.DataFrame({
        "Parameter": ["Temperature (Celsius)", "Relative Humidity (percentage)", "Wind Speed (km/hr)", "Cloud Cover (percentage)", "Pressure (milibars)"],
        "Value": [temp, relative_humidity, wind_speed, cloud_cover, pressure]
    })
    st.table(input_df)

st.sidebar.info("Adjust the sliders to see different predictions.")
st.caption("Developed using Streamlit and Scikit-learn")