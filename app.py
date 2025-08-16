import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

# Get feature names from the model (if available)
try:
    feature_names = model.feature_names_in_
except AttributeError:
    # Fallback if feature names aren't stored
    feature_names = [f"Feature {i+1}" for i in range(10)]

# Streamlit UI
st.title("🔍 ML Model Predictor")
st.write("Enter values for each feature below:")

# Collect user inputs
user_inputs = []
for name in feature_names:
    val = st.number_input(f"{name}", value=0.0)
    user_inputs.append(val)

# Predict button
if st.button("Predict"):
    input_array = np.array(user_inputs).reshape(1, -1)
    prediction = model.predict(input_array)
    st.success(f"✅ Prediction: {prediction[0]}")
