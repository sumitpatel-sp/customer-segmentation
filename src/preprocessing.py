from sklearn.preprocessing import StandardScaler

def preprocess(df):

    features = df[["Annual Income (k$)", "Spending Score (1-100)"]]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    return scaled_data, scaler
