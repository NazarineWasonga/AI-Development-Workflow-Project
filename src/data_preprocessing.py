import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """Load CSV data."""
    return pd.read_csv(path)

def preprocess_data(df):
    """Clean and preprocess data."""
    
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))

    # Separate features and labels
    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]

    # Normalize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

if __name__ == "__main__":
    df = load_data("../data/synthetic_patient_data.csv")
    X, y, scaler = preprocess_data(df)
    print("Preprocessing complete. Features shape:", X.shape)
