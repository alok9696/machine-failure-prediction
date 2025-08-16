import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import requests
import time

st.set_page_config(layout="wide")
st.title("🧠 Machine Health Dashboard")

# 🔄 Auto-refresh every 10 seconds
st.markdown("""
    <meta http-equiv="refresh" content="10">
""", unsafe_allow_html=True)

# Load model and training data
model = joblib.load("model.pkl")
df_train = pd.read_csv("sensor_data.csv")

# Fetch live sensor data
try:
    response = requests.get("http://YOUR_SERVER_IP:5000/latest")
    live_data = response.json()
    live_df = pd.DataFrame([live_data])
except:
    st.warning("⚠️ Could not fetch live sensor data")
    live_df = pd.DataFrame()

# Combine training + live data
df_combined = pd.concat([df_train, live_df], ignore_index=True)

# Sidebar: Select graph type
graph_type = st.sidebar.selectbox("Choose Graph Type", ["Distributions", "Pairplot", "Correlation Heatmap"])

# 📊 Distribution Plots
features = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

if graph_type == "Distributions":
    st.subheader("Feature Distributions")
    for feature in features:
        fig, ax = plt.subplots()
        sns.histplot(df_combined[feature], kde=True, ax=ax)
        st.pyplot(fig)

# 🔗 Pairplot
elif graph_type == "Pairplot":
    st.subheader("Pairwise Feature Relationships")
    fig = sns.pairplot(df_combined[features])
    st.pyplot(fig)

# 🔥 Correlation Heatmap
elif graph_type == "Correlation Heatmap":
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots()
    corr = df_combined[features].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# 🤖 Prediction
if not live_df.empty:
    input_array = live_df[features].values.reshape(1, -1)
    prediction = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0][1]

    st.subheader("🔍 Machine Status Prediction")
    if prediction == 1:
        st.error("🚨 NOW YOUR MACHINE IS GOING TO FAIL")
    elif proba > 0.7:
        st.warning("⚠️ THE MACHINE IS IN RISK")
    else:
        st.success("✅ THE MACHINE IS SAFE")

    st.write(f"Prediction Confidence (Failure): {proba*100:.2f}%")
    st.write(f"Live Timestamp: {live_data['timestamp']}")
