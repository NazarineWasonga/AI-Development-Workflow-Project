import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score
from model_training import train_model

def evaluate():
    """Evaluate model performance."""
    
    model, (X_test, y_test) = train_model()
    
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Evaluation Results:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

if __name__ == "__main__":
    evaluate()
