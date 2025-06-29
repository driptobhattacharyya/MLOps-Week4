import pytest
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train import load_data, train_model
from sklearn.metrics import accuracy_score

def test_data_shape():
    X_train, _, _, _ = load_data()
    assert X_train.shape[0] == 90, "Training data size incorrect"

def test_model_accuracy():
    # Train and save the model
    train_model()
    # Load the saved model
    loaded_model = joblib.load('artifacts/model.joblib')
    _, X_test, _, y_test = load_data()
    y_pred = loaded_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy >= 0.9, f"Model accuracy too low: {accuracy:.2f}"