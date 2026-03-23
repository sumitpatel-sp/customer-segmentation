import numpy as np


def predict_cluster(input_data, model, scaler) -> int:
    """
    Predict the KMeans cluster for a single customer.

    Parameters
    ----------
    input_data : list or array-like
        Raw (unscaled) RFM values: [Recency, Frequency, Monetary]
    model  : sklearn.cluster.KMeans — trained clustering model
    scaler : sklearn.preprocessing.StandardScaler — fitted on RFM features

    Returns
    -------
    int — cluster ID (0 to n_clusters-1)
    """
    data   = np.array([input_data])        # shape (1, 3)
    scaled = scaler.transform(data)
    return int(model.predict(scaled)[0])
