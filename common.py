%%writefile common.py
"""
common.py — Shared Input Validation, Normalization, and Feature Engineering.

This module serves as the single source of truth for data preprocessing across both
the offline training pipeline (train.py) and the live Streamlit inference interface
(app.py). By sharing this exact transformation pipeline, we prevent training-serving
skew and guarantee that model features remain identical during training and scoring[cite: 1].
"""

import numpy as np
import pandas as pd

# Core transaction columns expected from the IBM Transactions for AML dataset[cite: 1].
REQUIRED_COLUMNS = [
    "Timestamp",           # Date and time the transaction occurred
    "From Bank",           # Originating bank identifier
    "Account",             # Originating account identifier
    "To Bank",             # Beneficiary bank identifier
    "Account.1",           # Beneficiary account identifier (standardized from 'To Account' if needed)
    "Amount Received",     # Funds credited to the beneficiary
    "Receiving Currency",  # Currency of the received amount
    "Amount Paid",         # Funds debited from the originator
    "Payment Currency",    # Currency of the paid amount
    "Payment Format",      # Transfer channel (e.g., Wire, ACH, Cheque, Bitcoin)
]

# Ordered list of engineered numerical features passed into the ML classifier.
# CRITICAL: Do not alter this order without retraining the model artifact[cite: 1].
FEATURES = [
    "amount_paid",           # Log-transformed payment amount
    "amount_received",       # Log-transformed received amount
    "hour",                  # Hour of day (0-23) to capture off-hours activity
    "day_of_week",           # Day of week (0=Monday, 6=Sunday)
    "same_account",          # Flag (1/0) indicating self-transfers between identical accounts
    "same_bank",             # Flag (1/0) indicating intra-bank vs. inter-bank transfers
    "currency_mismatch",     # Flag (1/0) indicating cross-currency conversion
    "amount_difference",     # Log-transformed discrepancy between paid and received amounts
    "sender_prior_count",    # Cumulative historical transaction count for the sender
    "receiver_prior_count",  # Cumulative historical transaction count for the receiver
    "sender_prior_total",    # Log-transformed cumulative historical volume sent by originator
    "receiver_prior_total",  # Log-transformed cumulative historical volume received by beneficiary
    "payment_format_code",   # Integer encoding of the payment channel
]

# Deterministic mapping of payment channels to integer codes.
# Unknown or unseen payment formats safely default to 0.
PAYMENT_FORMAT_CODES = {
    "ACH": 1,
    "Cheque": 2,
    "Credit Card": 3,
    "Reinvestment": 4,
    "Wire": 5,
    "Cash": 6,
    "Bitcoin": 7,
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names and whitespace variations across IBM AML dataset variants.

    Different exports of the IBM AML dataset (e.g., HI-Small vs. LI-Small) sometimes
    use 'To Account' instead of 'Account.1' or include leading/trailing spaces.

    Args:
        df: Raw input DataFrame loaded from CSV or Excel.

    Returns:
        A DataFrame with cleaned, standardized column headers.
    """
    df = df.copy()
    # Strip accidental leading/trailing whitespace from all column headers
    df.columns = [str(col).strip() for col in df.columns]

    # Harmonize alternative column naming conventions[cite: 1]
    if "Account.1" not in df.columns and "To Account" in df.columns:
        df.rename(columns={"To Account": "Account.1"}, inplace=True)
    if "Account" not in df.columns and "From Account" in df.columns:
        df.rename(columns={"From Account": "Account"}, inplace=True)
    if "Is Laundering" not in df.columns and "Is_Laundering" in df.columns:
        df.rename(columns={"Is_Laundering": "Is Laundering"}, inplace=True)

    return df


def prepare(
    data: pd.DataFrame,
    inplace: bool = False,
    require_target: bool = False,
) -> pd.DataFrame:
    """
    Validate raw transaction records and engineer behavioral/transactional features[cite: 1].

    To prevent data leakage, all historical account aggregations (prior counts and
    prior totals) are computed strictly AFTER sorting chronologically and exclude
    the current transaction itself[cite: 1]. The target label ('Is Laundering') is never
    included in the feature matrix[cite: 1].

    Args:
        data: Raw transaction DataFrame.
        inplace: If True, modifies the DataFrame in place to conserve RAM on large files.
        require_target: If True, enforces presence of the 'Is Laundering' column (used during training).

    Returns:
        DataFrame enriched with all engineered columns defined in FEATURES.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Input must be a non-empty pandas DataFrame.")

    df = data if inplace else data.copy()
    df = normalize_columns(df)

    # 1. Schema Validation: Ensure all required columns exist before processing
    required = list(REQUIRED_COLUMNS)
    if require_target:
        required.append("Is Laundering")

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    # 2. Temporal Parsing & Chronological Ordering
    # Strict chronological sorting is essential so cumulative historical features
    # only reflect past activity[cite: 1].
    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df.dropna(subset=["Timestamp"], inplace=True)
    df.sort_values("Timestamp", kind="stable", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise ValueError("No rows contain a valid timestamp.")

    # 3. Categorical & Identifier Sanitization
    # Treat bank and account IDs as opaque strings to preserve leading zeros[cite: 1, 2].
    for col in ["From Bank", "Account", "To Bank", "Account.1"]:
        df[col] = (
            df[col]
            .astype(str)
            .replace({"nan": "UNKNOWN", "None": "UNKNOWN", "<NA>": "UNKNOWN"})
        )

    for col in ["Receiving Currency", "Payment Currency", "Payment Format"]:
        df[col] = (
            df[col]
            .astype(str)
            .replace({"nan": "UNKNOWN", "None": "UNKNOWN", "<NA>": "UNKNOWN"})
            .str.strip()
        )

    # 4. Monetary Sanitization & Log Scaling
    # Coerce invalid amounts to 0.0, clip negatives, and downcast to float32 to save 50% RAM.
    for col in ["Amount Paid", "Amount Received"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = (
            df[col]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .clip(lower=0)
            .astype("float32")
        )

    # Apply log1p (log(1 + x)) to compress extreme outliers while safely handling 0.0 values[cite: 1]
    df["amount_paid"] = np.log1p(df["Amount Paid"]).astype("float32")
    df["amount_received"] = np.log1p(df["Amount Received"]).astype("float32")

    # 5. Cyclical / Calendar Features
    df["hour"] = df["Timestamp"].dt.hour.astype("int8")
    df["day_of_week"] = df["Timestamp"].dt.dayofweek.astype("int8")

    # 6. Entity Relationship & Currency Indicators
    # Composite keys uniquely identify an account within its specific bank
    sender = df["From Bank"] + ":" + df["Account"]
    receiver = df["To Bank"] + ":" + df["Account.1"]

    df["same_account"] = (sender == receiver).astype("int8")
    df["same_bank"] = (df["From Bank"] == df["To Bank"]).astype("int8")
    df["currency_mismatch"] = (
        df["Receiving Currency"] != df["Payment Currency"]
    ).astype("int8")

    # Measure fee/spread discrepancies only when both sides use the same currency[cite: 1]
    df["amount_difference"] = np.where(
        df["currency_mismatch"] == 0,
        np.log1p((df["Amount Paid"] - df["Amount Received"]).abs()),
        0.0,
    ).astype("float32")

    # 7. Historical Velocity & Cumulative Volume Features (Leakage-Free)
    # cumcount() yields 0 for an account's first transaction, 1 for its second, etc.[cite: 1]
    df["sender_prior_count"] = df.groupby(sender, sort=False).cumcount().astype("int32")
    df["receiver_prior_count"] = df.groupby(receiver, sort=False).cumcount().astype("int32")

    # Subtract the current transaction's amount from cumsum() so only PRIOR volume is visible[cite: 1]
    sender_total = df.groupby(sender, sort=False)["Amount Paid"].cumsum()
    receiver_total = df.groupby(receiver, sort=False)["Amount Received"].cumsum()

    df["sender_prior_total"] = np.log1p(
        (sender_total - df["Amount Paid"]).clip(lower=0)
    ).astype("float32")
    df["receiver_prior_total"] = np.log1p(
        (receiver_total - df["Amount Received"]).clip(lower=0)
    ).astype("float32")

    # 8. Categorical Encoding & Final Finite-Value Guarantee
    df["payment_format_code"] = (
        df["Payment Format"].map(PAYMENT_FORMAT_CODES).fillna(0).astype("int8")
    )

    for feat in FEATURES:
        df[feat] = df[feat].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df
