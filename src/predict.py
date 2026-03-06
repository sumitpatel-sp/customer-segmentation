import numpy as np


def predict_cluster(input_data, model, scaler):
    """
    Predict the cluster for a single customer.

    Parameters
    ----------
    input_data : list or array-like
        Raw feature values in the order:
        [Gender (0/1), Age, Annual Income (k$), Spending Score (1-100)].
    model : sklearn.cluster.KMeans
        Trained clustering model.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler used during training (fitted on Income & Spending only).
    """
    # Model is trained on [Income, Spending] only
    income   = input_data[2]
    spending = input_data[3]
    data   = np.array([[income, spending]])
    scaled = scaler.transform(data)
    cluster = model.predict(scaled)
    return int(cluster[0])
