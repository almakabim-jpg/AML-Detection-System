
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from common import FEATURES, normalize_columns, prepare


# Resolve paths relative to this script so deployment works consistently.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "aml_model.joblib"
METRICS_PATH = BASE_DIR / "evaluation_metrics.json"


# Configure the Streamlit application.
st.set_page_config(
    page_title="AML Statement Analyzer",
    page_icon="🔎",
    layout="wide",
)


# Apply a simple light interface with consistent spacing and readable text.
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9fc;
        color: #172033;
    }

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e3e8ef;
        padding: 16px;
        border-radius: 10px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e3e8ef;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load the trained model once and reuse it during the application session.
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    try:
        artifact = joblib.load(MODEL_PATH)

        # Check that the saved artifact contains the required model information.
        if not isinstance(artifact, dict):
            return None

        if "model" not in artifact or "features" not in artifact:
            return None

        if list(artifact["features"]) != FEATURES:
            return None

        return artifact

    except Exception:
        return None


# Read the evaluation metrics saved during model training.
@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def read_uploaded_csv(uploaded_file):
    """Read and standardize a user-uploaded transaction CSV."""

    # Keep account and bank identifiers as strings to preserve their values.
    df = pd.read_csv(
        uploaded_file,
        dtype={
            "Account": "string",
            "Account.1": "string",
            "To Account": "string",
            "From Account": "string",
            "From Bank": "string",
            "To Bank": "string",
        },
        low_memory=False,
    )

    return normalize_columns(df)


def score_transactions(df, artifact, threshold):
    """Generate transaction-level risk scores using the trained classifier."""

    model = artifact["model"]
    features = artifact["features"]

    # Confirm that the uploaded data has all features expected by the model.
    missing_features = [
        feature for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: " + ", ".join(missing_features)
        )

    # Calculate the probability that each transaction belongs to the
    # laundering class.
    probabilities = model.predict_proba(
        df[features]
    )[:, 1]

    results = df.copy()

    results["Risk Score"] = probabilities
    results["Risk Score (%)"] = (
        results["Risk Score"] * 100
    ).round(2)

    # Apply the user's selected threshold to identify transactions
    # that should be reviewed by an analyst.
    results["Risk Level"] = np.where(
        results["Risk Score"] >= threshold,
        "Flagged",
        "Not Flagged",
    )

    return results


def build_account_summary(df):
    """Create descriptive account-level activity summaries."""

    # Represent each transaction from both the sender and receiver perspective.
    sent = df[
        ["Account", "Amount Paid", "Risk Score"]
    ].copy()

    sent.columns = [
        "Account",
        "Transaction Amount",
        "Risk Score",
    ]

    sent["Transaction Direction"] = "Sent"

    received = df[
        ["Account.1", "Amount Received", "Risk Score"]
    ].copy()

    received.columns = [
        "Account",
        "Transaction Amount",
        "Risk Score",
    ]

    received["Transaction Direction"] = "Received"

    # Combine sent and received transactions to summarize account activity.
    activity = pd.concat(
        [sent, received],
        ignore_index=True
    )

    summary = activity.groupby("Account").agg(
        Transaction_Count=("Transaction Amount", "count"),
        Total_Amount=("Transaction Amount", "sum"),
        Average_Amount=("Transaction Amount", "mean"),
        Maximum_Risk_Score=("Risk Score", "max"),
        Average_Risk_Score=("Risk Score", "mean"),
    ).reset_index()

    # Count the number of flagged transactions associated with each account.
    flagged = activity[
        activity["Risk Score"] >= st.session_state.get(
            "risk_threshold",
            0.5
        )
    ]

    flagged_counts = (
        flagged.groupby("Account")
        .size()
        .reset_index(name="Flagged_Transactions")
    )

    summary = summary.merge(
        flagged_counts,
        on="Account",
        how="left"
    )

    summary["Flagged_Transactions"] = (
        summary["Flagged_Transactions"]
        .fillna(0)
        .astype(int)
    )

    return summary.sort_values(
        ["Flagged_Transactions", "Total_Amount"],
        ascending=False
    )


def build_network_summary(df):
    """Summarize transaction relationships between sender and receiver accounts."""

    # Aggregate repeated transfers between the same account pairs.
    network = df.groupby(
        ["Account", "Account.1"],
        as_index=False
    ).agg(
        Transaction_Count=("Amount Paid", "count"),
        Total_Amount=("Amount Paid", "sum"),
        Maximum_Risk_Score=("Risk Score", "max"),
        Average_Risk_Score=("Risk Score", "mean"),
    )

    network = network.rename(
        columns={
            "Account": "Sender Account",
            "Account.1": "Receiver Account",
        }
    )

    return network.sort_values(
        ["Maximum_Risk_Score", "Total_Amount"],
        ascending=False
    )


def show_download_button(df, filename, label):
    """Provide a CSV download for the selected analysis results."""

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=label,
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


# Display the application title and explain its purpose.
st.title("AML Statement Analyzer")

st.write(
    "Analyze structured bank transaction data, identify potentially "
    "suspicious transactions, and review account and transaction "
    "relationships."
)

st.caption(
    "Automated risk scores are indicators for further investigation. "
    "They do not establish that money laundering has occurred."
)


# Load the trained model and its evaluation metrics.
artifact = load_model()
metrics = load_metrics()


# Let the user upload a structured transaction CSV.
st.sidebar.header("New Analysis")

uploaded_file = st.sidebar.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
)

# The threshold controls which model scores are classified as flagged.
threshold = st.sidebar.slider(
    "Flagging threshold",
    min_value=0.01,
    max_value=0.99,
    value=0.50,
    step=0.01,
)

st.session_state["risk_threshold"] = threshold


# Display model availability before the user starts an analysis.
if artifact is None:
    st.sidebar.warning(
        "No compatible trained model was found. "
        "Dataset profiling will still be available."
    )
else:
    st.sidebar.success("Trained model loaded.")


# Stop the analysis until a CSV file has been uploaded.
if uploaded_file is None:
    st.info(
        "Upload a transaction CSV using the sidebar to begin an analysis."
    )

    # Explain the required input format.
    with st.expander("Required CSV structure"):
        st.write(
            "The dataset should contain Timestamp, From Bank, Account, "
            "To Bank, Account.1, Amount Received, Receiving Currency, "
            "Amount Paid, Payment Currency, and Payment Format."
        )

    st.stop()


# Read and prepare the uploaded dataset.
try:
    raw_df = read_uploaded_csv(uploaded_file)

    # The uploaded dataset does not need laundering labels to be analyzed.
    prepared_df = prepare(
        raw_df,
        require_target=False
    )

except Exception as error:
    st.error(f"Could not process the uploaded dataset: {error}")
    st.stop()


if prepared_df.empty:
    st.error("The uploaded dataset contains no usable transactions.")
    st.stop()


# Add model risk scores when a compatible trained model is available.
if artifact is not None:
    try:
        results_df = score_transactions(
            prepared_df,
            artifact,
            threshold
        )
    except Exception as error:
        st.error(f"Could not score transactions: {error}")
        st.stop()
else:
    # Without a trained model, only descriptive dataset analysis is available.
    results_df = prepared_df.copy()
    results_df["Risk Score"] = np.nan
    results_df["Risk Score (%)"] = np.nan
    results_df["Risk Level"] = "Model unavailable"


# Calculate the main dashboard statistics.
total_transactions = len(results_df)
total_senders = results_df["Account"].nunique()
total_receivers = results_df["Account.1"].nunique()

if artifact is not None:
    flagged_transactions = int(
        (results_df["Risk Score"] >= threshold).sum()
    )

    flagged_percentage = (
        flagged_transactions / total_transactions * 100
    )
else:
    flagged_transactions = None
    flagged_percentage = None


# Create the main application navigation.
tabs = st.tabs(
    [
        "Dashboard",
        "Transactions",
        "Accounts",
        "Network",
        "Reports",
        "Model & Metrics",
    ]
)


# Dashboard: provide an overview of the uploaded transaction dataset.
with tabs[0]:
    st.header("Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Unique Senders",
        f"{total_senders:,}"
    )

    col3.metric(
        "Unique Receivers",
        f"{total_receivers:,}"
    )

    if artifact is not None:
        col4.metric(
            "Flagged Transactions",
            f"{flagged_transactions:,}",
            f"{flagged_percentage:.2f}% of transactions"
        )
    else:
        col4.metric(
            "Flagged Transactions",
            "Unavailable"
        )

    st.divider()

    # Display the distribution of transaction amounts.
    st.subheader("Transaction Amount Distribution")

    amount_data = results_df[
        ["Amount Paid", "Amount Received"]
    ].describe()

    st.dataframe(
        amount_data,
        use_container_width=True
    )

    # Show transaction activity over time when timestamps are available.
    st.subheader("Transaction Activity Over Time")

    daily_activity = (
        results_df
        .set_index("Timestamp")
        .resample("D")
        .size()
        .rename("Transactions")
    )

    st.line_chart(daily_activity)

    # Show the highest-risk transactions when a model is available.
    if artifact is not None:
        st.subheader("Highest-Risk Transactions")

        highest_risk = results_df.sort_values(
            "Risk Score",
            ascending=False
        ).head(10)

        st.dataframe(
            highest_risk,
            use_container_width=True
        )


# Transactions: allow analysts to inspect and filter individual transfers.
with tabs[1]:
    st.header("Transaction Analysis")

    if artifact is None:
        st.warning(
            "Transaction risk classification requires a trained model."
        )
    else:
        flagged_only = st.checkbox(
            "Show flagged transactions only"
        )

        transaction_view = results_df.copy()

        if flagged_only:
            transaction_view = transaction_view[
                transaction_view["Risk Score"] >= threshold
            ]

        st.write(
            f"Displaying {len(transaction_view):,} transactions."
        )

        st.dataframe(
            transaction_view,
            use_container_width=True,
            height=500,
        )

        show_download_button(
            transaction_view,
            "transaction_analysis.csv",
            "Download transaction results",
        )


# Accounts: summarize activity associated with sending and receiving accounts.
with tabs[2]:
    st.header("Account Analysis")

    if artifact is None:
        st.warning(
            "Account-level results are descriptive summaries only. "
            "Transaction risk scores require a trained model."
        )

    account_summary = build_account_summary(results_df)

    st.dataframe(
        account_summary,
        use_container_width=True,
        height=500,
    )

    # Allow an analyst to inspect the activity associated with one account.
    if not account_summary.empty:
        selected_account = st.selectbox(
            "Select an account",
            account_summary["Account"].astype(str).tolist(),
        )

        account_transactions = results_df[
            (
                results_df["Account"].astype(str) == selected_account
            )
            |
            (
                results_df["Account.1"].astype(str) == selected_account
            )
        ]

        st.subheader("Selected Account Transactions")

        st.dataframe(
            account_transactions,
            use_container_width=True,
        )

    show_download_button(
        account_summary,
        "account_analysis.csv",
        "Download account analysis",
    )


# Network: summarize which accounts send money to other accounts.
with tabs[3]:
    st.header("Transaction Network Analysis")

    st.write(
        "This view summarizes transaction relationships between accounts. "
        "It is a descriptive network analysis, not a separately trained "
        "network anomaly detection model."
    )

    network_summary = build_network_summary(results_df)

    st.dataframe(
        network_summary,
        use_container_width=True,
        height=500,
    )

    show_download_button(
        network_summary,
        "network_analysis.csv",
        "Download network analysis",
    )


# Reports: provide downloadable transaction, account, and network results.
with tabs[4]:
    st.header("Reports")

    st.write(
        "Export the results of the current analysis for further review."
    )

    show_download_button(
        results_df,
        "aml_transaction_report.csv",
        "Download full transaction report",
    )

    show_download_button(
        build_account_summary(results_df),
        "aml_account_report.csv",
        "Download account report",
    )

    show_download_button(
        build_network_summary(results_df),
        "aml_network_report.csv",
        "Download network report",
    )

    # Provide a compact summary that can be used in an investigation report.
    st.subheader("Analysis Summary")

    summary = {
        "Total Transactions": total_transactions,
        "Unique Sender Accounts": total_senders,
        "Unique Receiver Accounts": total_receivers,
        "Model Available": artifact is not None,
        "Flagging Threshold": threshold,
    }

    if artifact is not None:
        summary["Flagged Transactions"] = flagged_transactions
        summary["Flagged Percentage"] = round(
            flagged_percentage,
            2
        )

    st.json(summary)


# Model & Metrics: show the model information and its saved test evaluation.
with tabs[5]:
    st.header("Model & Evaluation Metrics")

    if artifact is None:
        st.warning(
            "No trained model is available. Train the model and place "
            "aml_model.joblib beside app.py."
        )
    else:
        st.success("A trained transaction-level model is loaded.")

        st.write(
            f"**Model:** {artifact.get('model_name', 'Unknown')}"
        )

        st.write(
            f"**Default threshold:** {artifact.get('threshold', 0.5)}"
        )

        if not artifact.get("threshold_calibrated", False):
            st.info(
                "The model threshold has not been calibrated. "
                "Choose a threshold based on validation results and "
                "the intended investigation workload."
            )

        st.write("**Model features:**")

        st.code(
            "\n".join(artifact["features"]),
            language="text"
        )

    st.divider()

    if metrics is None:
        st.info(
            "Evaluation metrics are not available. "
            "Run train.py to generate evaluation_metrics.json."
        )
    else:
        st.subheader("Test Set Performance")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        metric_col1.metric(
            "Average Precision",
            f"{metrics.get('average_precision', 0):.4f}"
        )

        metric_col2.metric(
            "ROC-AUC",
            f"{metrics.get('roc_auc', 0):.4f}"
        )

        metric_col3.metric(
            "F1-score",
            f"{metrics.get('f1_score', 0):.4f}"
        )

        metric_col4, metric_col5, metric_col6 = st.columns(3)

        metric_col4.metric(
            "Precision",
            f"{metrics.get('precision', 0):.4f}"
        )

        metric_col5.metric(
            "Recall",
            f"{metrics.get('recall', 0):.4f}"
        )

        metric_col6.metric(
            "Test Transactions",
            f"{metrics.get('test_rows', 0):,}"
        )

        # Display the full classification report when available.
        if "classification_report" in metrics:
            st.subheader("Classification Report")

            report_df = pd.DataFrame(
                metrics["classification_report"]
            ).transpose()

            st.dataframe(
                report_df,
                use_container_width=True
            )

        # Display the saved confusion matrix when it is available.
        confusion_path = BASE_DIR / "confusion_matrix.csv"

        if confusion_path.exists():
            st.subheader("Confusion Matrix")

            confusion_df = pd.read_csv(
                confusion_path,
                index_col=0
            )

            st.dataframe(
                confusion_df,
                use_container_width=True
            )
