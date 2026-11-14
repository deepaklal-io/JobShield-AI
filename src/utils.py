import joblib
import pandas as pd

from src.preprocessing import RAW_INPUT_COLUMNS, BINARY_FEATURES


def load_model(path):
    """Load the saved bundle and return (pipeline, threshold)."""
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle["threshold"]


def predict_posting(posting, pipeline, threshold):
    """Score ONE job posting given as a dict of raw fields.

    Missing keys and blank strings are treated as missing values.
    Returns a dict with the fraud probability and the final label.
    """
    values = {}
    for col in RAW_INPUT_COLUMNS:
        value = posting.get(col)
        if isinstance(value, str) and not value.strip():
            value = None
        if col in BINARY_FEATURES and value is None:
            value = 0
        values[col] = value

    row = pd.DataFrame([values])
    fraud_index = list(pipeline.classes_).index(1)
    probability = float(pipeline.predict_proba(row)[:, fraud_index][0])
    return {
        "fraud_probability": round(probability, 4),
        "is_fraud": bool(probability >= threshold),
        "threshold": threshold,
    }