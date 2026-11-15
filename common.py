"""
common.py — shared input validation and feature engineering.

Training and Streamlit prediction both call prepare(), so they use the
same feature definitions and transformations.
"""
import numpy as np
import pandas as pd

# IBM HI-Small transaction columns expected by this prototype.
REQUIRED_COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format",
]

# Keep this order identical during training and prediction.
FEATURES = [
    "amount_paid", "amount_received", "hour", "day_of_week",
    "same_account", "same_bank", "currency_mismatch", "amount_difference",
    "sender_prior_count", "receiver_prior_count",
    "sender_prior_total", "receiver_prior_total", "payment_format_code",
]

PAYMENT_FORMAT_CODES = {
    "ACH": 1, "Cheque": 2, "Credit Card": 3, "Reinvestment": 4,
    "Wire": 5, "Cash": 6, "Bitcoin": 7,
}


def prepare(data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Validate and transform transaction rows into model-ready features.

    Prior-activity fields are calculated after chronological sorting and
    exclude the current transaction. The target label is never a feature.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Input must be a non-empty pandas DataFrame.")

    df = data if inplace else data.copy()

    # Support an alternate receiving-account column name safely.
    if "Account.1" not in df.columns and "To Account" in df.columns:
        df.rename(columns={"To Account": "Account.1"}, inplace=True)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    # Parse timestamps, discard invalid dates, then preserve chronological order.
    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df.dropna(subset=["Timestamp"], inplace=True)
    df.sort_values("Timestamp", kind="stable", inplace=True)
    df.reset_index(drop=True, inplace=True)
    if df.empty:
        raise ValueError("No rows contain a valid timestamp.")

    # Identifiers are categorical strings, not numeric quantities.
    for col in ["From Bank", "Account", "To Bank", "Account.1"]:
        df[col] = df[col].astype(str).replace({"nan": "UNKNOWN", "None": "UNKNOWN"})

    # Normalize descriptive fields.
    for col in ["Receiving Currency", "Payment Currency", "Payment Format"]:
        df[col] = df[col].astype(str).replace({"nan": "UNKNOWN", "None": "UNKNOWN"}).str.strip()

    # Convert malformed or missing amounts to zero and prevent negatives.
    for col in ["Amount Paid", "Amount Received"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).astype("float32")

    # Log scaling reduces the influence of extremely large transaction values.
    df["amount_paid"] = np.log1p(df["Amount Paid"]).astype("float32")
    df["amount_received"] = np.log1p(df["Amount Received"]).astype("float32")

    # Extract time characteristics.
    df["hour"] = df["Timestamp"].dt.hour.astype("int8")
    df["day_of_week"] = df["Timestamp"].dt.dayofweek.astype("int8")

    # Create bank-account identifiers to analyze sender/receiver histories.
    sender = df["From Bank"] + ":" + df["Account"]
    receiver = df["To Bank"] + ":" + df["Account.1"]

    # Transaction relationship and currency indicators.
    df["same_account"] = (sender == receiver).astype("int8")
    df["same_bank"] = (df["From Bank"] == df["To Bank"]).astype("int8")
    df["currency_mismatch"] = (
        df["Receiving Currency"] != df["Payment Currency"]
    ).astype("int8")

    # Amounts in different currencies are not directly comparable.
    df["amount_difference"] = np.where(
        df["currency_mismatch"] == 0,
        np.log1p((df["Amount Paid"] - df["Amount Received"]).abs()),
        0.0,
    ).astype("float32")

    # Cumulative counts/totals represent activity before the current row.
    df["sender_prior_count"] = df.groupby(sender, sort=False).cumcount().astype("int32")
    df["receiver_prior_count"] = df.groupby(receiver, sort=False).cumcount().astype("int32")

    sender_total = df.groupby(sender, sort=False)["Amount Paid"].cumsum()
    receiver_total = df.groupby(receiver, sort=False)["Amount Received"].cumsum()
    df["sender_prior_total"] = np.log1p(
        (sender_total - df["Amount Paid"]).clip(lower=0)
    ).astype("float32")
    df["receiver_prior_total"] = np.log1p(
        (receiver_total - df["Amount Received"]).clip(lower=0)
    ).astype("float32")

    # Convert payment format labels into numeric categories.
    df["payment_format_code"] = (
        df["Payment Format"].map(PAYMENT_FORMAT_CODES).fillna(0).astype("int8")
    )

    # Guarantee finite numeric inputs for scikit-learn.
    for feat in FEATURES:
        df[feat] = df[feat].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df
