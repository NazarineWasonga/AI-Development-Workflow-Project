import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_data, preprocess_data

def train_model():
    """Train a Random Forest model on patient readmission data."""

    df = load_data("../data/synthetic_patient_data.csv")
    X, y, scaler = preprocess_data(df)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save model and scaler
    joblib.dump(model, "../model.pkl")
    joblib.dump(scaler, "../scaler.pkl")

    print("Model training complete.")
    return model, (X_test, y_test)

if __name__ == "__main__":
    train_model()
