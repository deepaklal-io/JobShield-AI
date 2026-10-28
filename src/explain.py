"""Explanations based on actual model evidence (shared by evaluation and the Flask app).

For a linear model (e.g. TF-IDF + Logistic Regression) the contribution of a feature to
one prediction is coefficient * feature value, which is exact for the log-odds.
For tree models only global feature importances exist, so direction is not reported.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer

from src.preprocessing import RAW_INPUT_COLUMNS, BINARY_FEATURES, CATEGORICAL_FEATURES


def posting_to_frame(posting):
    """Same row-building rules as src.utils.predict_posting."""
    values = {}
    for col in RAW_INPUT_COLUMNS:
        value = posting.get(col)
        if isinstance(value, str) and not value.strip():
            value = None
        if col in BINARY_FEATURES and value is None:
            value = 0
        values[col] = value
    return pd.DataFrame([values])


def _pretty(name):
    """'tfidf_description__free money' -> 'description: "free money"'."""
    prefix, _, rest = name.partition("__")
    if prefix.startswith("tfidf_"):
        return f'{prefix[6:]}: "{rest}"'
    if prefix == "categorical":
        for col in CATEGORICAL_FEATURES:
            if rest.startswith(col + "_"):
                return f"{col} = {rest[len(col) + 1:]}"
    return rest or name


def explain_frame(row, pipeline, top_n=8):
    """Return the top features pushing this posting toward fraud or toward legitimate."""
    classifier = pipeline.steps[-1][1]
    column_transformer = next(
        (s for _, s in pipeline.steps if isinstance(s, ColumnTransformer)), None)
    if column_transformer is None:
        return {"method": "unavailable", "toward_fraud": [], "toward_legit": []}

    names = column_transformer.get_feature_names_out()
    x = sparse.csr_matrix(pipeline[:-1].transform(row))

    if hasattr(classifier, "coef_"):
        contrib = np.asarray(x.multiply(classifier.coef_.ravel()).todense()).ravel()
        method = "linear_contribution"
    else:
        return {"method": "unavailable", "toward_fraud": [], "toward_legit": []}

    order = np.argsort(contrib)
    fraud = [(names[i], contrib[i]) for i in order[::-1][:top_n] if contrib[i] > 0]
    legit = [(names[i], contrib[i]) for i in order[:top_n] if contrib[i] < 0]
    fmt = lambda items: [{"feature": _pretty(n), "weight": round(float(w), 4)} for n, w in items]
    return {"method": method, "toward_fraud": fmt(fraud), "toward_legit": fmt(legit)}


def explain_posting(posting, pipeline, top_n=8):
    return explain_frame(posting_to_frame(posting), pipeline, top_n)


def global_top_features(pipeline, top_n=20):
    """Features with the largest overall weight toward fraud / toward legitimate."""
    classifier = pipeline.steps[-1][1]
    column_transformer = next(
        (s for _, s in pipeline.steps if isinstance(s, ColumnTransformer)), None)
    if column_transformer is None or not hasattr(classifier, "coef_"):
        return None
    names = column_transformer.get_feature_names_out()
    coef = classifier.coef_.ravel()
    order = np.argsort(coef)
    return pd.DataFrame({
        "toward_fraud": [_pretty(names[i]) for i in order[::-1][:top_n]],
        "fraud_weight": coef[order[::-1][:top_n]],
        "toward_legit": [_pretty(names[i]) for i in order[:top_n]],
        "legit_weight": coef[order[:top_n]],
    })