from sklearn.preprocessing import StandardScaler

def preprocess(df):

    df = df.drop("CustomerID", axis=1)
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

    features = df[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    return scaled_data, scaler
