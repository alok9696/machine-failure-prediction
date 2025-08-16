import streamlit as st
import joblib  # or pickle

# Load your trained model
model = joblib.load("model.pkl")

st.title("ML Model Predictor")

# Example input
user_input = st.number_input("Enter a number:")
prediction = model.predict([[user_input]])

st.write(f"Prediction: {prediction[0]}")
