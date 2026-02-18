import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from src.data_loader import load_data
from src.preprocessing import preprocess


def train_kmeans():

    # Load dataset
    df = load_data("data/mall_customers.csv")

    # Preprocess
    data, scaler = preprocess(df)

    mlflow.set_experiment("Consumer_Segmentation_ML")

    with mlflow.start_run():

        model = KMeans(n_clusters=5, random_state=42)
        model.fit(data)

        # Create models folder if not exists
        os.makedirs("models", exist_ok=True)

        # Save model
        joblib.dump(model, "models/kmeans.pkl")

        print("✅ KMeans model trained successfully")
        print("📁 Model saved at models/kmeans.pkl")


# 🔥 THIS IS THE IMPORTANT PART
if __name__ == "__main__":
    train_kmeans()

joblib.dump(scaler, "models/scaler.pkl")
