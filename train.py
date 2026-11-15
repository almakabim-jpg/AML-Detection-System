
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common import FEATURES, normalize_columns, prepare


# Keep all generated training artifacts in the same directory as this script.
OUTPUT_DIR = Path(__file__).resolve().parent

MODEL_PATH = OUTPUT_DIR / "aml_model.joblib"
METRICS_PATH = OUTPUT_DIR / "evaluation_metrics.json"
CONFUSION_PATH = OUTPUT_DIR / "confusion_matrix.csv"

# Limit the number of normal transactions used during training.
# This reduces memory usage while retaining all available laundering examples.
MAX_NORMAL_TRAIN = 250_000

# Limit evaluation size to avoid excessive memory consumption.
MAX_TEST_ROWS = 300_000


def load_dataset(dataset_path):
    """Load the IBM AML dataset and normalize supported column variations."""

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    # Read the header first so that only relevant columns are loaded.
    headers = pd.read_csv(
        dataset_path,
        nrows=0
    ).columns.tolist()

    headers = [str(column).strip() for column in headers]

    # Include alternative column names supported by common.py.
    supported_columns = {
        "Timestamp",
        "From Bank",
        "From Account",
        "Account",
        "To Bank",
        "To Account",
        "Account.1",
        "Amount Received",
        "Receiving Currency",
        "Amount Paid",
        "Payment Currency",
        "Payment Format",
        "Is Laundering",
        "Is_Laundering",
    }

    selected_columns = [
        column for column in headers
        if column in supported_columns
    ]

    if not selected_columns:
        raise ValueError(
            "No recognized IBM AML dataset columns were found."
        )

    # Load account identifiers as strings to avoid losing leading zeros.
    dtype_map = {
        column: "string"
        for column in selected_columns
        if column in {
            "Account",
            "Account.1",
            "To Account",
            "From Account",
            "From Bank",
            "To Bank",
        }
    }

    df = pd.read_csv(
        dataset_path,
        usecols=selected_columns,
        dtype=dtype_map,
        low_memory=False,
    )

    return normalize_columns(df)


def evaluate_model(model, X_test, y_test):
    """Calculate classification metrics for the held-out test transactions."""

    predictions = model.predict(X_test)

    # HistGradientBoosting provides probabilities for the positive class.
    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate ranking and classification metrics.
    metrics = {
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "test_rows": int(len(y_test)),
        "positive_test_rows": int(y_test.sum()),
        "negative_test_rows": int((y_test == 0).sum()),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    # Save the confusion matrix separately for easier reporting.
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=["Actual Normal", "Actual Laundering"],
        columns=["Predicted Normal", "Predicted Laundering"],
    )

    confusion_df.to_csv(CONFUSION_PATH)

    return metrics


def main():
    """Run the complete AML model training and evaluation pipeline."""

    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: python train.py /path/to/LI-Small_Trans.csv"
        )

    dataset_path = sys.argv[1]

    print("Loading AML dataset...")
    df = load_dataset(dataset_path)

    # The training dataset must contain the ground-truth laundering label.
    if "Is Laundering" not in df.columns:
        raise ValueError(
            "The dataset must contain the 'Is Laundering' target column."
        )

    print(f"Loaded {len(df):,} transactions.")

    # Remove records with missing or unsupported target labels.
    df["Is Laundering"] = pd.to_numeric(
        df["Is Laundering"],
        errors="coerce"
    )

    df = df[
        df["Is Laundering"].isin([0, 1])
    ].copy()

    if df.empty:
        raise ValueError(
            "No valid target labels were found in the dataset."
        )

    df["Is Laundering"] = df["Is Laundering"].astype(int)

    print("Preparing transaction features...")

    # Feature preparation calculates historical account activity
    # using transaction timestamps.
    df = prepare(
        df,
        require_target=True
    )

    # Sort chronologically so future transactions are not used to train
    # a model that is evaluated on earlier transactions.
    df = df.sort_values(
        "Timestamp",
        kind="stable"
    ).reset_index(drop=True)

    if df["Is Laundering"].nunique() < 2:
        raise ValueError(
            "Training requires both normal and laundering transactions."
        )

    # Use the earliest 80% of transactions for training and the latest
    # 20% for testing to provide a chronological evaluation.
    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    if train_df["Is Laundering"].nunique() < 2:
        raise ValueError(
            "The training portion must contain both target classes."
        )

    if test_df["Is Laundering"].nunique() < 2:
        raise ValueError(
            "The testing portion must contain both target classes."
        )

    print("Preparing training data...")

    # Keep every positive example and sample a manageable number
    # of normal transactions to reduce class imbalance and memory use.
    positive_train = train_df[
        train_df["Is Laundering"] == 1
    ]

    negative_train = train_df[
        train_df["Is Laundering"] == 0
    ]

    if len(negative_train) > MAX_NORMAL_TRAIN:
        negative_train = negative_train.sample(
            n=MAX_NORMAL_TRAIN,
            random_state=42
        )

    balanced_train = pd.concat(
        [positive_train, negative_train],
        ignore_index=True
    )

    # Shuffle the sampled training records before fitting the classifier.
    balanced_train = balanced_train.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # Sample the test set only when it exceeds the evaluation limit.
    # The chronological split itself remains unchanged.
    if len(test_df) > MAX_TEST_ROWS:
        test_df = test_df.sample(
            n=MAX_TEST_ROWS,
            random_state=42
        )

    X_train = balanced_train[FEATURES]
    y_train = balanced_train["Is Laundering"]

    X_test = test_df[FEATURES]
    y_test = test_df["Is Laundering"]

    print(f"Training transactions: {len(X_train):,}")
    print(f"Testing transactions: {len(X_test):,}")
    print(f"Laundering examples in training: {int(y_train.sum()):,}")

    # HistGradientBoosting is a CPU-based gradient boosting classifier.
    # Early stopping helps reduce unnecessary training iterations.
    model = HistGradientBoostingClassifier(
        max_iter=140,
        max_leaf_nodes=20,
        learning_rate=0.08,
        early_stopping=True,
        random_state=42,
    )

    print("Training AML classifier...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    # Save the trained model and the exact feature order required by app.py.
    # The 0.5 threshold is a starting point, not a calibrated AML decision threshold.
    model_artifact = {
        "model": model,
        "features": FEATURES,
        "threshold": 0.5,
        "model_name": "HistGradientBoostingClassifier",
        "threshold_calibrated": False,
    }

    joblib.dump(
        model_artifact,
        MODEL_PATH
    )

    # Save evaluation results in JSON format for the Streamlit dashboard.
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    print("\nTraining completed successfully.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_PATH}")

    print("\nEvaluation results:")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
