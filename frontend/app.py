import streamlit as st
import requests
import pandas as pd
import joblib

st.set_page_config(page_title="Consumer Segmentation", layout="wide")

st.title("🚀 Consumer Segmentation Dashboard")

API_URL = "http://127.0.0.1:8000"
USD_TO_INR = 83  # Fixed conversion rate

# =====================================================
# 🔹 LOAD DATA + MODEL FOR SEGMENT ANALYSIS
# =====================================================

df = pd.read_csv("data/mall_customers.csv")

model = joblib.load("models/kmeans.pkl")
scaler = joblib.load("models/scaler.pkl")

# Preprocess same as training
df_processed = df.copy()
df_processed = df_processed.drop("CustomerID", axis=1)
df_processed["Gender"] = df_processed["Gender"].map({"Male": 0, "Female": 1})

features = df_processed[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]
scaled = scaler.transform(features)

df["Cluster"] = model.predict(scaled)

# =====================================================
# 🔹 AUTOMATIC SEGMENT MEANING
# =====================================================

cluster_summary = df.groupby("Cluster")[[
    "Annual Income (k$)",
    "Spending Score (1-100)"
]].mean()

segment_meanings = {}

for cluster_id, row in cluster_summary.iterrows():

    income = row["Annual Income (k$)"]
    spending = row["Spending Score (1-100)"]

    if income < 50 and spending < 50:
        label = "Low Income - Low Spending"
    elif income >= 50 and spending >= 50:
        label = "High Income - High Spending"
    elif income >= 50 and spending < 50:
        label = "High Income - Low Spending"
    else:
        label = "Low Income - High Spending"

    segment_meanings[cluster_id] = label

# =====================================================
# 🔹 SECTION 1: SEGMENT INTERPRETATION
# =====================================================

st.header("📘 Segment Interpretation")

for cluster_id, meaning in segment_meanings.items():
    st.write(f"**Segment {cluster_id} → {meaning}**")

# =====================================================
# 🔹 SECTION 2: PREDICT NEW CUSTOMER (INR INPUT)
# =====================================================

st.header("🔍 Predict Customer Segment (Enter Income in ₹)")

gender = st.selectbox("Gender (0=Male, 1=Female)", [0, 1])
age = st.slider("Age", 18, 70)

income_inr = st.number_input("Annual Income (₹ INR)", min_value=10000, step=50000)

spending = st.slider("Spending Score (1-100)", 1, 100)

if st.button("Predict Segment"):

    try:
        # Convert INR → USD → k$
        usd = income_inr / USD_TO_INR
        income_k_dollar = usd / 1000

        response = requests.post(
            f"{API_URL}/predict",
            json={
                "Gender": gender,
                "Age": age,
                "Income": income_k_dollar,
                "Spending": spending
            }
        )

        cluster = response.json()["Cluster"]
        meaning = segment_meanings.get(cluster, "Unknown Segment")

        st.success(f"Predicted Segment: {cluster}")
        st.info(f"Segment Meaning: {meaning}")
        st.caption("Income converted internally from INR to USD for model prediction.")

    except:
        st.error("⚠️ Backend not running. Please start FastAPI.")
