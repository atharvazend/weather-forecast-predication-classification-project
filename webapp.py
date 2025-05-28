import streamlit as st
import pickle
import numpy as np

# Load the trained logistic regression model
with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)
with open("label_encoder.pkl", 'rb') as f:
    encoder = pickle.load(f)

# App title
st.title("🌦️ Rain Prediction Web App")
st.write("Enter the weather parameters below to predict whether it will rain or not.")

# Input form
with st.form("input_form"):
    temperature = st.number_input("🌡️ Temperature (°C) (0 to 50)", min_value=0.0, max_value=50.0 )
    humidity = st.number_input("💧 Relative Humidity (%) (0 to 100)", min_value=0.0, max_value=100.0)
    wind_speed = st.number_input("🌬️ Wind Speed (km/h) (0 to 150)", min_value=0.0, max_value=150.0)
    cloud_cover = st.number_input("☁️ Cloud Cover (%) (0 to 100)", min_value=0.0, max_value=100.0)
    pressure = st.number_input("📈 Pressure (millibars) (500 to 1500)", min_value=500.0, max_value=1500.0)

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
