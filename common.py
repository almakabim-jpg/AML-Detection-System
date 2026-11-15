
import numpy as np
import pandas as pd


# Standardize column names so the application can handle common IBM AML
# dataset variations without requiring manual changes to the CSV file.
COLUMN_ALIASES = {
    "To Account": "Account.1",
    "From Account": "Account",
    "Is_Laundering": "Is Laundering",
}


# These are the numerical features used by the transaction-level classifier.
FEATURES = [
    "amount_paid",
    "amount_received",
    "hour",
    "day_of_week",
    "same_account",
    "same_bank",
    "currency_mismatch",
    "amount_difference",
    "sender_prior_count",
    "receiver_prior_count",
    "sender_prior_total",
    "receiver_prior_total",
    "payment_format_code",
]


def normalize_columns(df):
    """Rename supported dataset columns to the application's standard names."""

    df = df.copy()

    # Remove accidental spaces around column names before applying aliases.
    df.columns = [str(column).strip() for column in df.columns]

    # Rename alternative column names only when the standard name is absent.
    rename_map = {
        old: new
        for old, new in COLUMN_ALIASES.items()
        if old in df.columns and new not in df.columns
    }

    return df.rename(columns=rename_map)


def validate_columns(df, require_target=False):
    """Check that the dataset contains the columns required for analysis."""

    required = [
        "Timestamp",
        "From Bank",
        "Account",
        "To Bank",
        "Account.1",
        "Amount Received",
        "Receiving Currency",
        "Amount Paid",
        "Payment Currency",
        "Payment Format",
    ]

    # Training requires the target label; uploaded analysis files do not.
    if require_target:
        required.append("Is Laundering")

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing)
        )


def prepare(df, require_target=False):
    """Clean transactions and calculate features for the AML classifier."""

    df = normalize_columns(df)
    validate_columns(df, require_target=require_target)

    # Work on a separate copy to avoid modifying the original dataframe.
    df = df.copy()

    # Preserve the original row order so results can be traced back to the CSV.
    df["_original_order"] = np.arange(len(df))

    # Convert timestamps and transaction amounts into usable data types.
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    df["Amount Paid"] = pd.to_numeric(
        df["Amount Paid"],
        errors="coerce"
    )

    df["Amount Received"] = pd.to_numeric(
        df["Amount Received"],
        errors="coerce"
    )

    # Remove records that cannot be used for chronological feature calculation.
    df = df.dropna(
        subset=[
            "Timestamp",
            "Account",
            "Account.1",
            "Amount Paid",
            "Amount Received",
        ]
    ).copy()

    # Account identifiers are treated as strings to preserve their values.
    for column in ["Account", "Account.1", "From Bank", "To Bank"]:
        df[column] = df[column].astype(str)

    # Sort transactions chronologically before calculating historical activity.
    df = df.sort_values(
        ["Timestamp", "_original_order"],
        kind="stable"
    ).reset_index(drop=True)

    # Basic transaction amount and time features.
    df["amount_paid"] = df["Amount Paid"].astype(float)
    df["amount_received"] = df["Amount Received"].astype(float)
    df["hour"] = df["Timestamp"].dt.hour
    df["day_of_week"] = df["Timestamp"].dt.dayofweek

    # Identify transactions where the sender and receiver are the same account.
    df["same_account"] = (
        df["Account"] == df["Account.1"]
    ).astype(int)

    # Compare the sending and receiving banks.
    df["same_bank"] = (
        df["From Bank"] == df["To Bank"]
    ).astype(int)

    # Different currencies can explain differences between paid and received amounts.
    df["currency_mismatch"] = (
        df["Payment Currency"] != df["Receiving Currency"]
    ).astype(int)

    # Calculate the amount difference only when both currencies match.
    df["amount_difference"] = np.where(
        df["currency_mismatch"] == 0,
        np.abs(df["Amount Paid"] - df["Amount Received"]),
        0.0
    )

    # Create a combined sender and receiver account key for historical calculations.
    sender = df["Account"]
    receiver = df["Account.1"]

    # Calculate the number of previous transactions made by each sender.
    # Subtracting one excludes the current transaction from its own history.
    df["sender_prior_count"] = (
        df.groupby(sender, sort=False)["Amount Paid"]
        .cumcount()
    )

    # Calculate the number of previous transactions received by each account.
    df["receiver_prior_count"] = (
        df.groupby(receiver, sort=False)["Amount Received"]
        .cumcount()
    )

    # Calculate cumulative sender amounts and subtract the current transaction.
    sender_cumulative = (
        df.groupby(sender, sort=False)["Amount Paid"]
        .cumsum()
    )

    df["sender_prior_total"] = (
        sender_cumulative - df["Amount Paid"]
    )

    # Calculate cumulative receiver amounts and subtract the current transaction.
    receiver_cumulative = (
        df.groupby(receiver, sort=False)["Amount Received"]
        .cumsum()
    )

    df["receiver_prior_total"] = (
        receiver_cumulative - df["Amount Received"]
    )

    # Convert payment formats into consistent numerical category codes.
    payment_format = (
        df["Payment Format"]
        .fillna("Unknown")
        .astype(str)
    )

    df["payment_format_code"] = (
        pd.Categorical(
            payment_format,
            categories=sorted(payment_format.unique())
        ).codes
    )

    # Replace invalid numerical values to prevent model prediction errors.
    df[FEATURES] = (
        df[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Restore the original CSV order after chronological calculations.
    df = (
        df.sort_values("_original_order", kind="stable")
        .drop(columns=["_original_order"])
        .reset_index(drop=True)
    )

    # Convert the target label to numeric values when it is available.
    if "Is Laundering" in df.columns:
        df["Is Laundering"] = pd.to_numeric(
            df["Is Laundering"],
            errors="coerce"
        )

    return df
