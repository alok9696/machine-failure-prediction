import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Machine Failure Predictor", layout="wide")

# Load historical sensor data
if os.path.exists("sensor_data.csv"):
    df_train = pd.read_csv("sensor_data.csv")
else:
    st.error("❌ 'sensor_data.csv' not found. Please upload it to the app directory.")
    st.stop()

# Sidebar navigation
option = st.sidebar.selectbox("Choose a view", ["5 Input", "Prediction"])

# --- 5 Input Tab ---
if option == "5 Input":
    st.header("🔧 Enter Sensor Values")

    sensor_1 = st.number_input("Sensor 1", value=0.0)
    sensor_2 = st.number_input("Sensor 2", value=0.0)
    sensor_3 = st.number_input("Sensor 3", value=0.0)
    sensor_4 = st.number_input("Sensor 4", value=0.0)
    sensor_5 = st.number_input("Sensor 5", value=0.0)

    input_data = pd.DataFrame([[sensor_1, sensor_2, sensor_3, sensor_4, sensor_5]],
                              columns=["Sensor_1", "Sensor_2", "Sensor_3", "Sensor_4", "Sensor_5"])

    st.subheader("📋 Your Input")
    st.dataframe(input_data)

# --- Prediction Tab ---
elif option == "Prediction":
    st.header("📊 Machine Failure Prediction")

    # Dummy model logic (replace with your actual model)
    def dummy_predict(data):
        threshold = df_train.mean()
        return int((data > threshold).sum().sum() > 2)

    # Use last input if available
    if "input_data" in locals():
        prediction = dummy_predict(input_data)
        result = "⚠️ Failure Predicted" if prediction == 1 else "✅ Normal Operation"
        st.subheader("🔮 Prediction Result")
        st.success(result if prediction == 0 else result)

        # Graphical comparison
        st.subheader("📈 Sensor Comparison with Historical Data")
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, col in enumerate(input_data.columns):
            ax.plot(df_train[col].values[:50], label=f"Past {col}", alpha=0.5)
            ax.plot(50, input_data[col].values[0], 'ro', label=f"Input {col}" if i == 0 else "")
        ax.set_title("Sensor Values: Input vs Historical")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Please enter sensor values in the '5 Input' tab first.")

