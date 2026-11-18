"""Explanations based on actual model evidence (shared by evaluation and the Flask app).

For a linear model (e.g. TF-IDF + Logistic Regression) the contribution of a feature to
one prediction is coefficient * feature value, which is exact for the log-odds.
For tree models only global feature importances exist, so direction is not reported.
"""
import numpy as np
import pandas as pd
from scipy import sparse
import re
from sklearn.compose import ColumnTransformer
from webcolors import names
from webcolors import names

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


def _pretty(name, salary_range=None, input_context=False):
    """Convert transformed feature names into readable labels."""
    prefix, _, rest = name.partition("__")

    salary_text = "" if salary_range is None else str(salary_range).strip()
    salary_missing = not salary_text or salary_text.lower() in {"nan", "none"}
    salary_parsed = bool(
        re.fullmatch(
            r"\s*\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*",
            salary_text
        )
    )

    if prefix.startswith("tfidf_"):
        return f'{prefix[6:]}: "{rest}"'

    if prefix == "categorical":
        for col in CATEGORICAL_FEATURES:
            if rest.startswith(col + "_"):
                value = rest[len(col) + 1:]
                if value == "Unknown":
                    return f"{col.replace('_', ' ')} not provided"
                return f"{col.replace('_', ' ')}: {value}"

    if prefix == "salary":
        labels = {
            "salary_min": "Minimum salary",
            "salary_max": "Maximum salary",
            "salary_missing": "Salary missing indicator",
            "salary_unparsed": "Salary format indicator",
        }

        if input_context and rest == "salary_missing":
            return (
                "Salary not provided indicator"
                if salary_missing
                else "Salary provided (missingness indicator)"
            )

        if input_context and rest == "salary_unparsed":
            if salary_missing:
                return "Salary format not applicable (salary missing)"
            return (
                "Salary format recognized indicator"
                if salary_parsed
                else "Salary format not recognized indicator"
            )

        return labels.get(rest, rest)

    if prefix == "binary":
        labels = {
            "telecommuting": "Remote work",
            "has_company_logo": "Company logo present",
            "has_questions": "Screening questions present",
        }
        return labels.get(rest, rest)

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
        if list(classifier.classes_) != [0, 1]:
            return {"method": "unavailable", "toward_fraud": [], "toward_legit": []}
        contrib = np.asarray(x.multiply(classifier.coef_.ravel()).todense()).ravel()
        method = "linear_contribution"
    else:
        return {"method": "unavailable", "toward_fraud": [], "toward_legit": []}

    # Identify which salary features are meaningful for this input.
    salary_range = row.iloc[0].get("salary_range")

    salary_text = (
        "" if salary_range is None or pd.isna(salary_range)
        else str(salary_range).strip()
    )

    salary_missing = not salary_text

    salary_parsed = bool(
        re.fullmatch(
            r"\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?",
            salary_text
        )
    )

    salary_unparsed = bool(salary_text) and not salary_parsed

    # Exclude inactive salary features from the explanation.
    eligible_indices = []

    for i, name in enumerate(names):
        if name == "salary__salary_min" or name == "salary__salary_max":
            if salary_parsed:
                eligible_indices.append(i)

        elif name == "salary__salary_missing":
            if salary_missing:
                eligible_indices.append(i)

        elif name == "salary__salary_unparsed":
            if salary_unparsed:
                eligible_indices.append(i)

        else:
            eligible_indices.append(i)

    # Rank only eligible features.
    order = sorted(
        eligible_indices,
        key=lambda i: contrib[i]
    )

    fraud = [
        (names[i], contrib[i])
        for i in order[::-1]
        if contrib[i] > 0
    ][:top_n]

    legit = [
        (names[i], contrib[i])
        for i in order
        if contrib[i] < 0
    ][:top_n]

    fmt = lambda items: [
        {
            "feature": _pretty(n, salary_range, input_context=True),
            "weight": round(float(w), 4)
        }
        for n, w in items
    ]

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