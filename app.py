import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import os

# --- Page Config ---
st.set_page_config(page_title="Machine Failure Predictor", layout="wide")
st.title("🛠️ Machine Failure Prediction Dashboard")

# --- Load Data ---
if os.path.exists("sensor_data.csv"):
    df_train = pd.read_csv("sensor_data.csv")
else:
    st.error("❌ 'sensor_data.csv' not found.")
    st.stop()

# --- Load Model ---
if os.path.exists("model.pkl"):
    model = joblib.load("model.pkl")
else:
    st.error("❌ 'model.pkl' not found.")
    st.stop()

# --- Sensor Input ---
st.header("🔧 Enter Sensor Values")

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

# --- Prediction ---
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

# --- Matplotlib Graphs ---
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

# --- Seaborn Visualizations ---
st.header("📊 Sensor Feature Distributions")

selected_feature = st.selectbox("Choose a feature to explore", 
    ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"])

fig, ax = plt.subplots()
sns.histplot(df_train[selected_feature], kde=True, ax=ax, color="skyblue")
ax.set_title(f"Distribution of {selected_feature}")
st.pyplot(fig)

# --- Correlation Heatmap ---
st.header("🔥 Feature Correlation Heatmap")

corr_matrix = df_train[["Air temperature [K]", "Process temperature [K]", 
                        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure"]].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# --- Failure Type Breakdown ---
st.header("⚙️ Failure Type Breakdown")

failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]
failure_counts = df_train[failure_types].sum()

fig, ax = plt.subplots()
sns.barplot(x=failure_types, y=failure_counts.values, palette="viridis", ax=ax)
ax.set_title("Failure Type Frequency")
ax.set_ylabel("Count")
st.pyplot(fig)

# --- Optional Pairplot ---
st.header("🧬 Pairplot of Sensor Features (May Take Time)")

if st.button("Generate Pairplot"):
    # Sample fewer points to reduce clutter
    sample_df = df_train.sample(n=200, random_state=42)

    # Create pairplot with transparency and corner layout
    pairplot = sns.pairplot(
        sample_df,
        vars=["Air temperature [K]", "Process temperature [K]", 
              "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"],
        hue="Machine failure",
        palette="Set2",
        plot_kws={'alpha': 0.6},
        corner=True
    )

    # Increase figure size for better spacing
   # pairplot.figure.set_size_inches(12, 12)

    # Adjust spacing between subplots
    pairplot.figure.subplots_adjust(hspace=0.4, wspace=0.4)

    # Display in Streamlit
    st.pyplot(pairplot.figure)
