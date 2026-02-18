import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from src.train_ml import train_kmeans

st.title("Consumer Segmentation Dashboard")

# Load dataset for visualization
df = pd.read_csv("data/mall_customers.csv")

st.subheader("Raw Data")
st.write(df.head())

# Scatter Plot
st.subheader("Cluster Visualization")

fig, ax = plt.subplots()
ax.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"])
ax.set_xlabel("Income")
ax.set_ylabel("Spending Score")
st.pyplot(fig)

@app.post("/retrain")
def retrain_model():
    train_kmeans()
    return {"message": "Model retrained successfully"}