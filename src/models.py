import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def create_pipeline(vectorizer, model_type: str = "logistic_regression"):
    """
    Creates a pipeline combining a vectorizer and a classifier.
    """
    if model_type == "logistic_regression":
        classifier = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    elif model_type == "naive_bayes":
        classifier = MultinomialNB(alpha=1.0)
    elif model_type == "svm":
        # Calibrate LinearSVC to support predict_proba for confidence scores
        classifier = CalibratedClassifierCV(LinearSVC(random_state=42, C=0.5))
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", classifier)
    ])

def train_and_evaluate_model(pipeline, X_train, y_train, X_val, y_val):
    """
    Trains the pipeline on the training set and evaluates it on the validation set.
    Returns metrics dict and confusion matrix.
    """
    print(f"Training model...")
    pipeline.fit(X_train, y_train)
    
    # Predict on validation set
    y_pred = pipeline.predict(X_val)
    
    # Evaluate metrics
    acc = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="weighted")
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)
    
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "classes": pipeline.classes_.tolist()
    }
    
    return pipeline, metrics

def save_model_pipeline(pipeline, model_name: str):
    """Saves the trained pipeline to the models/ directory."""
    path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")
    return path

def load_model_pipeline(model_name: str):
    """Loads a trained pipeline from the models/ directory."""
    path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    pipeline = joblib.load(path)
    print(f"Model loaded from {path}")
    return pipeline
