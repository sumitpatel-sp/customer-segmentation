import joblib
import numpy as np

def predict_cluster(input_data):

    model = joblib.load("models/kmeans.pkl")

    input_data = np.array(input_data).reshape(1, -1)
    cluster = model.predict(input_data)

    return int(cluster[0])
