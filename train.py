%%writefile train.py
"""
train.py — Offline Model Training and Chronological Evaluation Pipeline.

This script loads a labeled IBM AML transaction CSV, performs feature engineering
via common.py, splits the timeline chronologically (80% train / 20% test) to prevent
future-to-past data leakage, addresses severe class imbalance via negative undersampling,
trains a HistGradientBoostingClassifier, and exports the model artifact and evaluation
reports for the Streamlit application[cite: 2].

Usage:
    python train.py /path/to/HI-Small_Trans.csv
"""

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

# Save all output artifacts in the same directory as train.py so app.py locates them automatically[cite: 2]
OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = OUTPUT_DIR / "aml_model.joblib"
METRICS_PATH = OUTPUT_DIR / "evaluation_metrics.json"
CONFUSION_PATH = OUTPUT_DIR / "confusion_matrix.csv"

# Memory and Class-Imbalance Controls:
# AML datasets are heavily imbalanced (~0.1% positive). Retaining 100% of positive
# laundering cases while capping normal training transactions at 250,000 keeps RAM
# usage low and improves minority-class gradient updates[cite: 2].
MAX_NORMAL_TRAIN = 250_000
MAX_TEST_ROWS = 300_000


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """
    Load the IBM AML CSV efficiently by filtering columns and enforcing string IDs[cite: 2].

    Args:
        dataset_path: File system path to the labeled transaction CSV.

    Returns:
        Normalized DataFrame containing recognized AML columns.
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Inspect only the header row first to identify available columns without loading data
    headers = pd.read_csv(dataset_path, nrows=0).columns.tolist()
    headers = [str(column).strip() for column in headers]

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

    selected_columns = [col for col in headers if col in supported_columns]
    if not selected_columns:
        raise ValueError("No recognized IBM AML dataset columns were found.")

    # Force bank and account identifiers to load as strings so leading zeros are preserved[cite: 2]
    dtype_map = {
        col: "string"
        for col in selected_columns
        if col in {"Account", "Account.1", "To Account", "From Account", "From Bank", "To Bank"}
    }

    df = pd.read_csv(
        dataset_path,
        usecols=selected_columns,
        dtype=dtype_map,
        low_memory=False,
    )

    return normalize_columns(df)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the trained classifier on held-out future transactions and save reports[cite: 2].

    Because money laundering is rare, PR-AUC (average_precision) is tracked alongside
    ROC-AUC, precision, recall, and F1-score[cite: 2].

    Args:
        model: Fitted scikit-learn classifier.
        X_test: Engineered feature matrix for the test partition.
        y_test: Ground-truth binary labels (0 = Normal, 1 = Laundering).

    Returns:
        Dictionary of summary evaluation metrics.
    """
    predictions = model.predict(X_test)
    # Extract predicted probability for the positive class (1 = Laundering)[cite: 2]
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
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

    # Export a labeled confusion matrix CSV for easy inclusion in academic reports[cite: 2]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    confusion_df = pd.DataFrame(
        matrix,
        index=["Actual Normal", "Actual Laundering"],
        columns=["Predicted Normal", "Predicted Laundering"],
    )
    confusion_df.to_csv(CONFUSION_PATH)

    return metrics


def main():
    """Execute the end-to-end AML training, evaluation, and artifact export workflow[cite: 2]."""
    if len(sys.argv) < 2:
        raise ValueError("Usage: python train.py /path/to/HI-Small_Trans.csv")

    dataset_path = sys.argv[1]

    # Step 1: Load and validate ground-truth labels
    print("Loading AML dataset...")
    df = load_dataset(dataset_path)

    if "Is Laundering" not in df.columns:
        raise ValueError("The dataset must contain the 'Is Laundering' target column.")

    print(f"Loaded {len(df):,} transactions.")

    df["Is Laundering"] = pd.to_numeric(df["Is Laundering"], errors="coerce")
    df = df[df["Is Laundering"].isin([0, 1])].copy()

    if df.empty:
        raise ValueError("No valid target labels were found in the dataset.")

    df["Is Laundering"] = df["Is Laundering"].astype(int)

    # Step 2: Engineer behavioral and transactional features[cite: 2]
    print("Preparing transaction features...")
    df = prepare(df, require_target=True)
    df = df.sort_values("Timestamp", kind="stable").reset_index(drop=True)

    if df["Is Laundering"].nunique() < 2:
        raise ValueError("Training requires both normal and laundering transactions.")

    # Step 3: Chronological Train/Test Split (80% Past -> 20% Future)[cite: 2]
    # Splitting by time rather than randomly simulates real-world deployment where
    # historical transactions are used to detect future suspicious activity[cite: 2].
    split_index = int(len(df) * 0.80)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    if train_df["Is Laundering"].nunique() < 2 or test_df["Is Laundering"].nunique() < 2:
        raise ValueError("Both training and testing portions must contain both target classes.")

    # Step 4: Downsample Majority Class in Training Data[cite: 2]
    print("Preparing training data...")
    positive_train = train_df[train_df["Is Laundering"] == 1]
    negative_train = train_df[train_df["Is Laundering"] == 0]

    if len(negative_train) > MAX_NORMAL_TRAIN:
        negative_train = negative_train.sample(n=MAX_NORMAL_TRAIN, random_state=42)

    balanced_train = pd.concat([positive_train, negative_train], ignore_index=True)
    balanced_train = balanced_train.sample(frac=1, random_state=42).reset_index(drop=True)

    # Cap test evaluation size if needed while preserving the chronological boundary[cite: 2]
    if len(test_df) > MAX_TEST_ROWS:
        test_df = test_df.sample(n=MAX_TEST_ROWS, random_state=42)

    X_train = balanced_train[FEATURES]
    y_train = balanced_train["Is Laundering"]
    X_test = test_df[FEATURES]
    y_test = test_df["Is Laundering"]

    print(f"Training transactions: {len(X_train):,}")
    print(f"Testing transactions: {len(X_test):,}")
    print(f"Laundering examples in training: {int(y_train.sum()):,}")

    # Step 5: Fit HistGradientBoostingClassifier[cite: 2]
    # Histogram-based gradient boosting scales efficiently to hundreds of thousands
    # of rows and captures non-linear interactions across transaction features.
    model = HistGradientBoostingClassifier(
        max_iter=140,
        max_leaf_nodes=20,
        learning_rate=0.08,
        early_stopping=True,
        random_state=42,
    )

    print("Training AML classifier...")
    model.fit(X_train, y_train)

    # Step 6: Evaluate on Held-Out Future Transactions[cite: 2]
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    # Step 7: Serialize Model Artifact and Evaluation Metrics[cite: 2]
    # Bundle the exact feature list with the model so app.py can verify schema compatibility.
    model_artifact = {
        "model": model,
        "features": FEATURES,
        "threshold": 0.5,
        "model_name": "HistGradientBoostingClassifier",
        "threshold_calibrated": False,
    }

    joblib.dump(model_artifact, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

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


  
