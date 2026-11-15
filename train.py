"""
train.py — train and evaluate the baseline AML classifier.

Usage:
    python train.py "C:/path/to/HI-Small_Trans.csv"

Outputs are saved in the same folder as this script.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from common import FEATURES, prepare

# Reproducibility and training configuration.
RANDOM_STATE = 42
TRAIN_FRACTION = 0.80
MAX_TRAINING_NEGATIVES = 250_000

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "aml_model.joblib"
METRICS_PATH = BASE_DIR / "evaluation_metrics.json"
MATRIX_PATH = BASE_DIR / "confusion_matrix.csv"


def get_dataset_path() -> Path:
    """Read and validate the CSV path supplied on the command line."""
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python train.py "path/to/HI-Small_Trans.csv"')
    path = Path(sys.argv[1]).expanduser()
    if not path.is_file():
        raise SystemExit(f"Dataset file not found: {path}")
    return path


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the labelled IBM AML CSV and report its actual class counts."""
    print(f"Reading {path} (large files may take several minutes)...")
    df = pd.read_csv(
        path,
        dtype={"From Bank": str, "To Bank": str,
               "Account": str, "Account.1": str},
        low_memory=False,
    )
    if "Is Laundering" not in df.columns:
        raise SystemExit('Required target column "Is Laundering" was not found.')

    df["Is Laundering"] = pd.to_numeric(
        df["Is Laundering"], errors="coerce"
    )
    df = df[df["Is Laundering"].isin([0, 1])].copy()
    df["Is Laundering"] = df["Is Laundering"].astype(int)

    if df.empty or df["Is Laundering"].nunique() != 2:
        counts = df["Is Laundering"].value_counts().to_dict()
        raise SystemExit(
            f"Both labels 0 and 1 are required; found: {counts}. "
            "Verify that this is the complete labelled transaction CSV."
        )

    print("Label counts (0=normal, 1=laundering):",
          df["Is Laundering"].value_counts().sort_index().to_dict())
    return df


def chronological_split(df: pd.DataFrame):
    """Preprocess then reserve the latest 20% of rows for testing."""
    print("Cleaning data and engineering transaction/account features...")
    prepared = prepare(df)
    cut = int(len(prepared) * TRAIN_FRACTION)
    train_df = prepared.iloc[:cut].copy()
    test_df = prepared.iloc[cut:].copy()

    for name, part in [("training", train_df), ("test", test_df)]:
        if part["Is Laundering"].nunique() != 2:
            raise SystemExit(
                f"The chronological {name} partition does not contain both "
                "classes. Inspect label distribution over time."
            )
    print(f"Training rows: {len(train_df):,}; test rows: {len(test_df):,}")
    return train_df, test_df


def sample_training_rows(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep every positive training example and sample normal examples.
    The test set is not sampled, preserving its original class distribution.
    """
    positives = train_df[train_df["Is Laundering"] == 1]
    negatives = train_df[train_df["Is Laundering"] == 0]
    sampled_negatives = negatives.sample(
        n=min(MAX_TRAINING_NEGATIVES, len(negatives)),
        random_state=RANDOM_STATE,
    )
    sample = pd.concat([positives, sampled_negatives]).sample(
        frac=1, random_state=RANDOM_STATE
    )
    print(f"Training sample: {len(sample):,} rows "
          f"({len(positives):,} positive; {len(sampled_negatives):,} negative)")
    return sample


def fit_model(sample: pd.DataFrame):
    """Fit a supervised tree-based gradient boosting classifier."""
    model = HistGradientBoostingClassifier(
        max_iter=140,
        max_leaf_nodes=20,
        learning_rate=0.08,
        random_state=RANDOM_STATE,
    )
    print("Fitting HistGradientBoostingClassifier...")
    model.fit(sample[FEATURES], sample["Is Laundering"])
    return model


def evaluate(model, test_df: pd.DataFrame) -> dict:
    """Calculate threshold-based and ranking metrics on untouched test rows."""
    y_true = test_df["Is Laundering"].astype(int)
    scores = model.predict_proba(test_df[FEATURES])[:, 1]
    threshold = 0.50
    predictions = (scores >= threshold).astype(int)

    results = {
        "test_transactions": int(len(test_df)),
        "test_laundering_cases": int(y_true.sum()),
        "test_prevalence": float(y_true.mean()),
        "decision_threshold": threshold,
        "pr_auc_average_precision": float(
            average_precision_score(y_true, scores)
        ),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
    }

    # Save a labelled confusion matrix for reporting and presentation.
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual normal (0)", "Actual laundering (1)"],
        columns=["Predicted normal (0)", "Predicted laundering (1)"],
    )
    matrix_df.to_csv(MATRIX_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nEvaluation metrics:")
    for name, value in results.items():
        print(f"{name}: {value}")
    print("\nConfusion matrix:\n", matrix_df)
    print("\nClassification report:\n", classification_report(
        y_true, predictions, target_names=["Normal", "Laundering"],
        zero_division=0
    ))
    return results


def save_model(model):
    """Save the trained model and metadata for app.py to load."""
    joblib.dump({
        "model": model,
        "features": FEATURES,
        "threshold": 0.50,
        "model_name": "HistGradientBoostingClassifier",
    }, MODEL_PATH)
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved confusion matrix: {MATRIX_PATH}")
    print(
        "Caution: training uses negative-class undersampling. Model outputs "
        "are uncalibrated ranking scores, not verified laundering probabilities."
    )


def main():
    """Run dataset loading, preprocessing, training, evaluation, and saving."""
    dataset = load_dataset(get_dataset_path())
    train_df, test_df = chronological_split(dataset)
    training_sample = sample_training_rows(train_df)
    model = fit_model(training_sample)
    evaluate(model, test_df)
    save_model(model)


if __name__ == "__main__":
    main()
