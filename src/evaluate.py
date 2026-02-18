from sklearn.metrics import silhouette_score

def evaluate_model(model, data):
    labels = model.predict(data)
    score = silhouette_score(data, labels)
    print("Silhouette Score:", score)
