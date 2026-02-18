import shap
import joblib
import pandas as pd

model = joblib.load("models/kmeans.pkl")
scaler = joblib.load("models/scaler.pkl")

def explain_sample(sample):
    explainer = shap.KernelExplainer(model.predict, sample)
    shap_values = explainer.shap_values(sample)
    return shap_values
