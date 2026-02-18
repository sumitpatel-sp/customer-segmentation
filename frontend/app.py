import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.title("🚀 Consumer Segmentation System")

API_URL = "http://127.0.0.1:8000"

# ==============================
# Section 1: Prediction
# ==============================

st.header("🔍 Predict Customer Segment")

gender = st.selectbox("Gender", [0, 1])
age = st.slider("Age", 18, 70)
income = st.slider("Annual Income (k$)", 10, 150)
spending = st.slider("Spending Score (1-100)", 1, 100)

if st.button("Predict Segment"):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "Gender": gender,
                "Age": age,
                "Income": income,
                "Spending": spending
            }
        )
        cluster = response.json()["Cluster"]
        st.success(f"Customer belongs to Segment {cluster}")
    except:
        st.error("⚠️ FastAPI backend is not running!")

# ==============================
# Section 2: Retrain Model
# ==============================

st.header("🔁 Retrain Model")

if st.button("Retrain Model"):
    try:
        response = requests.post(f"{API_URL}/retrain")
        st.success(response.json()["message"])
    except:
        st.error("⚠️ Could not retrain. Backend not running.")

# ==============================
# Section 3: Visualization
# ==============================

st.header("📊 Cluster Visualization")

df = pd.read_csv("data/mall_customers.csv")

fig, ax = plt.subplots()
ax.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"])
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
st.pyplot(fig)
