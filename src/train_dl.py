import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
import mlflow.tensorflow
import os

def train_autoencoder(data):

    mlflow.set_experiment("Consumer_Segmentation_DL")

    with mlflow.start_run():

        input_dim = data.shape[1]

        input_layer = layers.Input(shape=(input_dim,))
        encoder = layers.Dense(16, activation="relu")(input_layer)
        encoder = layers.Dense(8, activation="relu")(encoder)

        decoder = layers.Dense(16, activation="relu")(encoder)
        decoder = layers.Dense(input_dim, activation="linear")(decoder)

        autoencoder = models.Model(input_layer, decoder)

        autoencoder.compile(optimizer="adam", loss="mse")

        autoencoder.fit(data, data, epochs=50, batch_size=16, verbose=1)

        os.makedirs("models", exist_ok=True)
        autoencoder.save("models/autoencoder.h5")

        mlflow.tensorflow.log_model(autoencoder, "autoencoder_model")

        print("Autoencoder trained successfully")
