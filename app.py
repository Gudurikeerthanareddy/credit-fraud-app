import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# Load model and scaler
@st.cache_resource
def load_files():
    try:
        with open("fraud_model_updated.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler_updated.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        return None, None

model, scaler = load_files()

# Title
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details below to check if it's fraudulent.")

# Input fields
amt = st.number_input("Transaction Amount", min_value=0.0, value=2000.0)
city_pop = st.number_input("City Population", min_value=0.0, value=1000.0)
unix_time = st.number_input("Unix Time", min_value=0.0, value=1500.0)

# Predict button
if st.button("Predict"):
    if model is None or scaler is None:
        st.warning("Model not loaded properly.")
    else:
        try:
            # Input array
            input_data = np.array([[amt, city_pop, unix_time]])

            # Scale input
            input_scaled = scaler.transform(input_data)

            # Prediction
            prediction = model.predict(input_scaled)[0]

            # Probability (if available)
            probability = None
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(input_scaled)[0][1]

            # Output
            if prediction == 1:
                st.error("🚨 Fraudulent Transaction")
            else:
                st.success("✅ Legitimate Transaction")

            if probability is not None:
                st.info(f"Fraud Probability: {probability:.4f}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")