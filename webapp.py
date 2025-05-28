import streamlit as st
import pickle
import numpy as np

# Load the trained logistic regression model
with open(r"C:\My folder\weather forecast predication classification project\logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)
with open(r"C:\My folder\weather forecast predication classification project\scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)
with open(r"C:\My folder\weather forecast predication classification project\label_encoder.pkl", 'rb') as f:
    encoder = pickle.load(f)

# App title
st.title("🌦️ Rain Prediction Web App")
st.write("Enter the weather parameters below to predict whether it will rain or not.")

# Input form
with st.form("input_form"):
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0 )
    humidity = st.number_input("💧 Relative Humidity (%)", min_value=0.0, max_value=100.0)
    wind_speed = st.number_input("🌬️ Wind Speed (km/h)", min_value=0.0, max_value=150.0)
    cloud_cover = st.number_input("☁️ Cloud Cover (%)", min_value=0.0, max_value=100.0)
    pressure = st.number_input("📈 Pressure (hPa)", min_value=900.0, max_value=1100.0)

    submit = st.form_submit_button("Predict Rain")

if submit:
    # Prepare the input for prediction
    input_data = np.array([[temperature, humidity, wind_speed, cloud_cover, pressure]])
    scaled_input_data = scaler.transform(input_data)

    # Predict using the logistic regression model
    prediction = model.predict(scaled_input_data)
    

    # Show the result
    if prediction[0] == 1:
        st.success(f"✅ It is likely to rain. 🌧️")
    else:
        st.info(f"❌ It is unlikely to rain. ☀️")
