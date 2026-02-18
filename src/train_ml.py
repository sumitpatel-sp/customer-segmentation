import os
import joblib
from sklearn.cluster import KMeans
from src.data_loader import load_data
from src.preprocessing import preprocess


def train_kmeans():

    # Load dataset
    df = load_data("data/mall_customers.csv")

    # Preprocess
    data, scaler = preprocess(df)

    # Train model
    model = KMeans(n_clusters=5, random_state=42)
    model.fit(data)

    # Create models folder
    os.makedirs("models", exist_ok=True)

    # Save model
    joblib.dump(model, "models/kmeans.pkl")

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    print("✅ Model and scaler saved successfully")


if __name__ == "__main__":
    train_kmeans()
