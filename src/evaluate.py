"""Final evaluation of the approved JobShield-AI model on the untouched test set.

Run from the project root:
    python -m src.evaluate                     (auto-finds the saved model)
    python -m src.evaluate --model path/to/model.joblib
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.explain import explain_frame, global_top_features  # noqa: E402
from src.utils import load_model  # noqa: E402

TARGET = "fraudulent"
REPORTS = ROOT / "reports"


def find_model():
    found = [p for ext in ("*.joblib", "*.pkl") for p in ROOT.rglob(ext)
             if ".git" not in p.parts]
    if not found:
        sys.exit("No .joblib/.pkl found. Pass --model path/to/model.joblib")
    if len(found) > 1:
        print("Several model files found, using the first:", *found, sep="\n  ")
    return found[0]


def get_test_split():
    """Identical to the team's split: dropna target, drop job_id, dedupe, 70/15/15, seed 42."""
    df = pd.read_csv(ROOT / "data" / "raw" / "fake_job_postings.csv")
    df = df.loc[df[TARGET].isin([0, 1])].copy()
    df = df.drop(columns=["job_id"]).drop_duplicates().reset_index(drop=True)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    return X_test, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()

    REPORTS.mkdir(exist_ok=True)
    model_path = args.model or find_model()
    pipeline, threshold = load_model(model_path)
    print("Model:", model_path)
    print("Pipeline steps:", [name for name, _ in pipeline.steps])
    print("Decision threshold (fixed on validation):", threshold)

    X_test, y_test = get_test_split()
    print("Test set:", X_test.shape, "fraud cases:", int(y_test.sum()))

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    metrics = pd.DataFrame([{
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "pr_auc": average_precision_score(y_test, proba),
        "threshold": threshold,
        "n_test": len(y_test),
    }]).round(4)
    metrics.to_csv(REPORTS / "test_metrics.csv", index=False)
    print("\n", metrics.T.to_string(header=False))

    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTN={tn}  FP={fp}  FN={fn}  TP={tp}")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Legit", "Fraud"])
    ax.set_yticks([0, 1], ["Legit", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    for (i, j), v in pd.DataFrame(cm).stack().items():
        ax.text(j, i, int(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=150)

    # Error analysis: save every FP / FN with its probability
    errors = X_test.copy()
    errors["actual"], errors["predicted"], errors["fraud_probability"] = y_test, pred, proba
    cols = ["title", "location", "employment_type", "company_profile", "description"]
    for label, mask in (("false_positives", (pred == 1) & (y_test == 0)),
                        ("false_negatives", (pred == 0) & (y_test == 1))):
        out = errors.loc[mask.values]
        out = out.assign(description=out["description"].fillna("").str[:300])
        out[cols + ["fraud_probability"]].sort_values(
            "fraud_probability", ascending=(label == "false_positives")
        ).to_csv(REPORTS / f"{label}.csv", index=False)
        print(f"Saved {label}.csv ({int(mask.sum())} rows)")

    # Explanations from real model evidence
    top = global_top_features(pipeline)
    if top is not None:
        top.to_csv(REPORTS / "global_top_features.csv", index=False)
        print("\nTop global features:\n", top.head(10).to_string(index=False))
    else:
        print("\nGlobal explanation skipped: final estimator has no coef_.")

    fn_rows = X_test[((pred == 0) & (y_test == 1)).values]
    if len(fn_rows):
        print("\nWhy the first missed fraud case was scored low:")
        print(explain_frame(fn_rows.iloc[[0]], pipeline))

    print("\nDone. Outputs are in", REPORTS)


if __name__ == "__main__":
    main()