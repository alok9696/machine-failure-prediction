import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

# Define feature names
feature_names = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

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
    prediction = model.predict(input_array)[0]

    # If model supports probability
    try:
        proba = model.predict_proba(input_array)[0][1]  # Probability of failure
    except:
        proba = None

    # Custom messages
    if prediction == 1:
        st.error("🚨 NOW YOUR MACHINE IS GOING TO FAIL")
    else:
        if proba is not None:
            if proba > 0.7:
                st.warning("⚠️ THE MACHINE IS IN RISK")
            else:
                st.success("✅ THE MACHINE IS SAFE")
        else:
            st.success("✅ THE MACHINE IS SAFE")

    if proba is not None:
        st.write(f"Prediction Confidence (Failure): {proba*100:.2f}%")
