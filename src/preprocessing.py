from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess(df):
    df = df.drop("CustomerID", axis=1)

    df["Gender"] = df["Gender"].map({"Male":0, "Female":1})

    features = df[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    return scaled, scaler
