import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
#import os
import json
import threading
import paho.mqtt.client as mqtt
import plotly.express as px

# --- Page Config ---
st.set_page_config(page_title="Smart Machine Dashboard", layout="wide")
st.title("🧠 Smart Machine Monitoring & Prediction")

# --- Sidebar Navigation ---
page = st.sidebar.radio("Choose Dashboard", ["📡 Live Sensor Dashboard", "🛠️ Failure Prediction"])

# --- MQTT Setup ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "esp8266/sensor/data"
sensor_data = []

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        sensor_data.append(payload)
    except Exception as e:
        print("Error parsing MQTT message:", e)

def start_mqtt():
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(MQTT_TOPIC)
    client.loop_forever()

if "mqtt_started" not in st.session_state:
    threading.Thread(target=start_mqtt, daemon=True).start()
    st.session_state["mqtt_started"] = True

# --- Page 1: Live Sensor Dashboard ---
if page == "📡 Live Sensor Dashboard":
    st.header("📡 ESP8266 Live Sensor Dashboard")

    if sensor_data:
        df = pd.DataFrame(sensor_data).drop_duplicates().tail(100)
        st.success("✅ Live data received from ESP8266")
    else:
        st.warning("⚠️ No live data yet. Showing fallback.")
        if os.path.exists("static_sensor_data.csv"):
            df = pd.read_csv("static_sensor_data.csv").tail(100)
        else:
            st.error("❌ 'static_sensor_data.csv' not found.")
            st.stop()

    st.subheader("📋 Latest Sensor Reading")
    st.write(df.iloc[-1])

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(df, y="temperature", title="Temperature Over Time", markers=True)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.line(df, y="humidity", title="Humidity Over Time", markers=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🔍 Sensor Feature Relationships")
    if st.button("Generate Scatter Matrix"):
        fig_matrix = px.scatter_matrix(
            df,
            dimensions=["temperature", "humidity"],
            title="Sensor Feature Relationships",
            color_discrete_sequence=["green"],
            height=600
        )
        st.plotly_chart(fig_matrix, use_container_width=True)

# --- Page 2: Machine Failure Prediction ---
elif page == "🛠️ Failure Prediction":
    st.header("🛠️ Machine Failure Prediction Dashboard")

    if os.path.exists("sensor_data.csv"):
        df_train = pd.read_csv("sensor_data.csv")
    else:
        st.error("❌ 'sensor_data.csv' not found.")
        st.stop()

    if os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")
    else:
        st.error("❌ 'model.pkl' not found.")
        st.stop()

    st.subheader("🔧 Enter Sensor Values")
    col1, col2, col3 = st.columns(3)
    with col1:
        air_temp = st.number_input("Air temperature [K]", value=300.0)
    with col2:
        process_temp = st.number_input("Process temperature [K]", value=310.0)
    with col3:
        rpm = st.number_input("Rotational speed [rpm]", value=1500.0)

    col4, col5 = st.columns(2)
    with col4:
        torque = st.number_input("Torque [Nm]", value=40.0)
    with col5:
        tool_wear = st.number_input("Tool wear [min]", value=20.0)

    input_data = pd.DataFrame([[air_temp, process_temp, rpm, torque, tool_wear]],
                              columns=["Air temperature [K]", "Process temperature [K]",
                                       "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"])

    st.subheader("📋 Your Input")
    st.dataframe(input_data)

    st.subheader("🔮 Prediction Result")
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("🚨 NOW YOUR MACHINE IS GOING TO FAIL")
    elif proba > 0.7:
        st.warning("⚠️ THE MACHINE IS IN RISK")
    else:
        st.success("✅ THE MACHINE IS SAFE")

    st.write(f"Prediction Confidence (Failure): {proba*100:.2f}%")

    st.header("📈 Sensor Graphs")
    def plot_feature(feature_name, label):
        fig, ax = plt.subplots()
        ax.plot(df_train[feature_name].values[:100], label="Historical")
        ax.plot(100, input_data[feature_name].values[0], 'ro', label="Your Input")
        ax.set_title(f"{label} Graph")
        ax.set_xlabel("Time")
        ax.set_ylabel(label)
        ax.legend()
        st.pyplot(fig)

    plot_feature("Air temperature [K]", "Air Temperature")
    plot_feature("Process temperature [K]", "Process Temperature")
    plot_feature("Rotational speed [rpm]", "RPM")
    plot_feature("Torque [Nm]", "Torque")
    plot_feature("Tool wear [min]", "Tool Wear")

    st.header("📊 Sensor Feature Distributions")
    selected_feature = st.selectbox("Choose a feature to explore", 
        ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"])

    fig, ax = plt.subplots()
    sns.histplot(df_train[selected_feature], kde=True, ax=ax, color="skyblue")
    ax.set_title(f"Distribution of {selected_feature}")
    st.pyplot(fig)

    st.header("🔥 Feature Correlation Heatmap")
    corr_matrix = df_train[["Air temperature [K]", "Process temperature [K]", 
                            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure"]].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.header("⚙️ Failure Type Breakdown")
    failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    failure_counts = df_train[failure_types].sum()

    fig, ax = plt.subplots()
    sns.barplot(x=failure_types, y=failure_counts.values, palette="viridis", ax=ax)
    ax.set_title("Failure Type Frequency")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.header("🧬 Pairplot of Sensor Features (May Take Time)")
    if st.button("Generate Pairplot"):
        sample_df = df_train.sample(n=200, random_state=42)
        features = ["Air temperature [K]", "Process temperature [K]", 
                    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
        pairplot = sns.pairplot(
            sample_df,
            vars=features,
            hue="Machine failure",
            palette="Set2",
            plot_kws={'alpha': 0.6}
        )
        pairplot.figure.set_size_inches(14, 14)
        for i, row in enumerate(features):
            for j, col in enumerate(features):
                ax = pairplot.axes[i][j]
                if ax is not None:
                    ax.set_xlabel(col, fontsize=9)
                    ax.set_ylabel(row, fontsize=9)
                    ax.tick_params(labelbottom=True, labelleft=True)
        pairplot.figure.tight_layout()
        st.pyplot(pairplot.figure)

