
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from common import prepare, FEATURES


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "aml_model.joblib"
METRICS_PATH = BASE_DIR / "evaluation_metrics.json"

st.set_page_config(
    page_title="AML Statement Analyzer",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LIGHT DASHBOARD STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F7F9FC;
        color: #172033;
    }

    h1, h2, h3 {
        color: #172033;
    }

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #E2E8F0;
        padding: 16px;
        border-radius: 10px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
    }

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL AND METRICS LOADING
# ============================================================

@st.cache_resource
def load_model():
    """Load the saved AML model without stopping the application."""

    if not MODEL_PATH.exists():
        return None

    try:
        artifact = joblib.load(MODEL_PATH)

        # Verify that the model artifact has the expected structure.
        if not isinstance(artifact, dict):
            return None

        if "model" not in artifact:
            return None

        if "features" not in artifact:
            return None

        # Ensure the saved feature order matches common.py.
        if list(artifact["features"]) != list(FEATURES):
            return None

        return artifact

    except Exception as error:
        st.warning(f"Could not load the trained model: {error}")
        return None


@st.cache_data
def load_metrics():
    """Load evaluation metrics when the training script has generated them."""

    if not METRICS_PATH.exists():
        return None

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


artifact = load_model()
metrics = load_metrics()


# ============================================================
# DATA PROCESSING FUNCTIONS
# ============================================================

def read_uploaded_csv(uploaded_file):
    """Read the uploaded transaction history while preserving account IDs."""

    return pd.read_csv(
        uploaded_file,
        dtype={
            "From Bank": str,
            "To Bank": str,
            "Account": str,
            "Account.1": str,
            "To Account": str,
        },
        low_memory=False,
    )


def analyze_transactions(raw_df, model_artifact, threshold):
    """Prepare transaction features and calculate risk scores if possible."""

    # Use the shared feature-engineering function used during model training.
    clean_df = prepare(raw_df)

    if clean_df.empty:
        raise ValueError("No valid transactions remain after data preparation.")

    # Do not attempt classification when no model is available.
    if model_artifact is None:
        clean_df["Risk Score"] = np.nan
        clean_df["Risk Level"] = "Model unavailable"
        clean_df["Flag for Review"] = False
        return clean_df

    # Verify that every feature required by the model exists.
    missing = [
        feature for feature in model_artifact["features"]
        if feature not in clean_df.columns
    ]

    if missing:
        raise ValueError(
            "Missing model features: " + ", ".join(missing)
        )

    # Generate a score for each transaction using the trained classifier.
    probabilities = model_artifact["model"].predict_proba(
        clean_df[model_artifact["features"]]
    )[:, 1]

    clean_df["Risk Score"] = probabilities
    clean_df["Risk Score (%)"] = (
        probabilities * 100
    ).round(2)

    # Flag transactions based on the selected investigation threshold.
    clean_df["Flag for Review"] = (
        clean_df["Risk Score"] >= threshold
    )

    clean_df["Risk Level"] = np.where(
        clean_df["Flag for Review"],
        "Flagged",
        "Not Flagged",
    )

    return clean_df


def create_account_summary(df):
    """Summarize the transaction activity of sender and receiver accounts."""

    # Build a sender activity table.
    sent = df[
        ["Account", "Amount Paid"]
    ].copy()

    sent.columns = ["Account", "Amount"]
    sent["Direction"] = "Sent"

    # Build a receiver activity table.
    received = df[
        ["Account.1", "Amount Received"]
    ].copy()

    received.columns = ["Account", "Amount"]
    received["Direction"] = "Received"

    # Combine both directions into one account activity table.
    activity = pd.concat(
        [sent, received],
        ignore_index=True
    )

    summary = activity.groupby("Account").agg(
        Transaction_Count=("Amount", "count"),
        Total_Amount=("Amount", "sum"),
        Average_Amount=("Amount", "mean"),
    ).reset_index()

    # Add risk information when model predictions are available.
    if "Risk Score" in df.columns and df["Risk Score"].notna().any():

        sent_risk = df[
            ["Account", "Risk Score", "Flag for Review"]
        ].copy()

        sent_risk.columns = [
            "Account",
            "Risk Score",
            "Flag for Review",
        ]

        received_risk = df[
            ["Account.1", "Risk Score", "Flag for Review"]
        ].copy()

        received_risk.columns = [
            "Account",
            "Risk Score",
            "Flag for Review",
        ]

        risk_activity = pd.concat(
            [sent_risk, received_risk],
            ignore_index=True
        )

        risk_summary = risk_activity.groupby("Account").agg(
            Maximum_Risk_Score=("Risk Score", "max"),
            Average_Risk_Score=("Risk Score", "mean"),
            Flagged_Transactions=("Flag for Review", "sum"),
        ).reset_index()

        summary = summary.merge(
            risk_summary,
            on="Account",
            how="left"
        )

    return summary.sort_values(
        "Total_Amount",
        ascending=False
    )


def create_network_summary(df):
    """Summarize the transfer relationships between account pairs."""

    network = df.groupby(
        ["From Bank", "Account", "To Bank", "Account.1"],
        dropna=False
    ).agg(
        Transaction_Count=("Amount Paid", "size"),
        Total_Amount=("Amount Paid", "sum"),
        Average_Amount=("Amount Paid", "mean"),
    ).reset_index()

    # Add risk information when model predictions are available.
    if "Risk Score" in df.columns and df["Risk Score"].notna().any():

        risk_summary = df.groupby(
            ["From Bank", "Account", "To Bank", "Account.1"],
            dropna=False
        ).agg(
            Maximum_Risk_Score=("Risk Score", "max"),
            Average_Risk_Score=("Risk Score", "mean"),
            Flagged_Transactions=("Flag for Review", "sum"),
        ).reset_index()

        network = network.merge(
            risk_summary,
            on=["From Bank", "Account", "To Bank", "Account.1"],
            how="left"
        )

    return network.sort_values(
        "Total_Amount",
        ascending=False
    )


def download_csv(df, filename, label):
    """Create a CSV download for an analysis result."""

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("AML Statement Analyzer")

st.markdown(
    """
    Analyze banking transaction histories, identify potentially
    suspicious transactions, and examine account activity and
    transaction relationships.
    """
)

st.caption(
    "Academic prototype • Transaction-level machine learning "
    "• Flags require human review"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("AML Analysis")

st.sidebar.markdown("### Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload transaction history CSV",
    type=["csv"],
)

st.sidebar.divider()

# The model threshold determines which transactions are flagged.
if artifact is not None:
    default_threshold = float(
        artifact.get("threshold", 0.5)
    )
else:
    default_threshold = 0.5

threshold = st.sidebar.slider(
    "Risk Review Threshold",
    min_value=0.0,
    max_value=1.0,
    value=default_threshold,
    step=0.05,
)

st.sidebar.divider()

# Give the user a clear indication of model availability.
if artifact is not None:
    st.sidebar.success("Trained model loaded")
else:
    st.sidebar.warning("Model not available")

st.sidebar.caption(
    "Only structured CSV transaction histories are supported."
)


# ============================================================
# INITIAL PAGE STATE
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a transaction-history CSV using the sidebar "
        "to begin your AML analysis."
    )

    # Explain what the application can analyze.
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Transaction Analysis")
        st.write(
            "Review individual transfers and identify transactions "
            "that may require further investigation."
        )

    with col2:
        st.subheader("Account Analysis")
        st.write(
            "Examine transaction counts, amounts, and activity "
            "associated with individual accounts."
        )

    with col3:
        st.subheader("Network Analysis")
        st.write(
            "Explore relationships between sending and receiving "
            "accounts."
        )

    st.divider()

    st.subheader("Supported Dataset")

    st.write(
        "The application expects an IBM AML-style transaction CSV "
        "containing the following fields:"
    )

    st.code(
        "Timestamp, From Bank, Account, To Bank, Account.1, "
        "Amount Received, Receiving Currency, Amount Paid, "
        "Payment Currency, Payment Format",
        language="text",
    )

    st.caption(
        "The application does not currently accept PDF bank statements."
    )

    st.stop()


# ============================================================
# LOAD AND ANALYZE UPLOADED DATA
# ============================================================

try:

    with st.spinner("Processing transaction history..."):

        raw_df = read_uploaded_csv(uploaded_file)

        if raw_df.empty:
            raise ValueError("The uploaded CSV file is empty.")

        results = analyze_transactions(
            raw_df,
            artifact,
            threshold
        )

except Exception as error:

    st.error(f"Unable to analyze the uploaded dataset: {error}")

    st.info(
        "Check that your CSV contains the required IBM AML transaction "
        "columns and that the file is not corrupted."
    )

    st.stop()


# ============================================================
# CALCULATE DASHBOARD STATISTICS
# ============================================================

total_transactions = len(results)

unique_senders = results["Account"].nunique()

unique_receivers = results["Account.1"].nunique()

total_banks = pd.concat(
    [results["From Bank"], results["To Bank"]]
).nunique()

model_available = (
    artifact is not None
    and results["Risk Score"].notna().any()
)

if model_available:

    flagged_transactions = int(
        results["Flag for Review"].sum()
    )

    flagged_percentage = (
        flagged_transactions / total_transactions
    ) * 100

else:

    flagged_transactions = None
    flagged_percentage = None


# ============================================================
# MAIN NAVIGATION
# ============================================================

dashboard_tab, transaction_tab, account_tab, network_tab, reports_tab, metrics_tab = st.tabs(
    [
        "Dashboard",
        "Transaction Analysis",
        "Account Analysis",
        "Network Analysis",
        "Reports",
        "Model & Metrics",
    ]
)


# ============================================================
# 1. DASHBOARD
# ============================================================

with dashboard_tab:

    st.header("Dashboard Overview")

    # Display the key statistics for the uploaded transaction dataset.
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Unique Senders",
        f"{unique_senders:,}"
    )

    col3.metric(
        "Unique Receivers",
        f"{unique_receivers:,}"
    )

    if model_available:

        col4.metric(
            "Flagged Transactions",
            f"{flagged_transactions:,}",
            f"{flagged_percentage:.2f}%"
        )

    else:

        col4.metric(
            "Flagged Transactions",
            "N/A"
        )

    st.divider()

    # Display a warning when the application is running without a model.
    if not model_available:

        st.warning(
            "The dataset was loaded successfully, but no trained model "
            "is available. Descriptive analysis is still enabled."
        )

    # Show the distribution of transaction amounts.
    st.subheader("Transaction Amount Statistics")

    amount_stats = results[
        ["Amount Paid", "Amount Received"]
    ].describe()

    st.dataframe(
        amount_stats,
        use_container_width=True
    )

    # Show the transaction count over time.
    st.subheader("Transaction Activity Over Time")

    daily_activity = (
        results
        .set_index("Timestamp")
        .resample("D")
        .size()
        .rename("Transactions")
    )

    st.line_chart(daily_activity)

    # Display the highest-risk transactions when a model is available.
    if model_available:

        st.subheader("Highest-Risk Transactions")

        highest_risk = results.sort_values(
            "Risk Score",
            ascending=False
        ).head(10)

        display_columns = [
            "Timestamp",
            "Account",
            "Account.1",
            "Amount Paid",
            "Risk Score",
            "Risk Level",
        ]

        st.dataframe(
            highest_risk[display_columns],
            use_container_width=True
        )


# ============================================================
# 2. TRANSACTION ANALYSIS
# ============================================================

with transaction_tab:

    st.header("Transaction-Level Analysis")

    if not model_available:

        st.warning(
            "Transaction-level risk classification requires a trained model."
        )

        st.dataframe(
            results.head(100),
            use_container_width=True
        )

    else:

        st.write(
            "Review individual transactions and their model-generated "
            "risk scores."
        )

        # Let the user focus on flagged transactions.
        show_flagged = st.checkbox(
            "Show flagged transactions only"
        )

        transaction_results = results.copy()

        if show_flagged:

            transaction_results = transaction_results[
                transaction_results["Flag for Review"]
            ]

        st.metric(
            "Transactions Displayed",
            f"{len(transaction_results):,}"
        )

        transaction_columns = [
            "Timestamp",
            "From Bank",
            "Account",
            "To Bank",
            "Account.1",
            "Amount Paid",
            "Payment Currency",
            "Payment Format",
            "Risk Score (%)",
            "Risk Level",
        ]

        st.dataframe(
            transaction_results[transaction_columns],
            use_container_width=True,
            height=500
        )

        download_csv(
            transaction_results,
            "aml_transaction_analysis.csv",
            "Download Transaction Analysis"
        )


# ============================================================
# 3. ACCOUNT ANALYSIS
# ============================================================

with account_tab:

    st.header("Account-Level Analysis")

    st.write(
        "Examine transaction activity associated with individual accounts."
    )

    account_results = create_account_summary(results)

    st.dataframe(
        account_results,
        use_container_width=True,
        height=450
    )

    st.divider()

    # Allow the user to inspect a specific account's transaction history.
    st.subheader("Individual Account Investigation")

    account_options = sorted(
        set(results["Account"].astype(str))
        | set(results["Account.1"].astype(str))
    )

    if account_options:

        selected_account = st.selectbox(
            "Select Account",
            account_options
        )

        account_transactions = results[
            (
                results["Account"].astype(str) == selected_account
            )
            |
            (
                results["Account.1"].astype(str) == selected_account
            )
        ]

        st.write(
            f"Transactions associated with account: {selected_account}"
        )

        st.dataframe(
            account_transactions,
            use_container_width=True
        )

    download_csv(
        account_results,
        "aml_account_analysis.csv",
        "Download Account Analysis"
    )


# ============================================================
# 4. NETWORK ANALYSIS
# ============================================================

with network_tab:

    st.header("Transaction Network Analysis")

    st.write(
        "Explore the relationships between sending and receiving accounts "
        "through their transaction activity."
    )

    st.info(
        "This is a descriptive account-relationship analysis. "
        "It is not a separately trained graph-based AML detection model."
    )

    network_results = create_network_summary(results)

    st.subheader("Account Connections")

    st.dataframe(
        network_results,
        use_container_width=True,
        height=500
    )

    # Highlight the most frequently used account connections.
    st.subheader("Most Frequent Account Connections")

    frequent_connections = network_results.sort_values(
        "Transaction_Count",
        ascending=False
    ).head(10)

    st.dataframe(
        frequent_connections,
        use_container_width=True
    )

    download_csv(
        network_results,
        "aml_network_analysis.csv",
        "Download Network Analysis"
    )


# ============================================================
# 5. REPORTS AND EXPORTS
# ============================================================

with reports_tab:

    st.header("Analysis Reports")

    st.write(
        "Export the transaction, account, and network analysis results."
    )

    st.subheader("Transaction Report")

    download_csv(
        results,
        "aml_full_transaction_report.csv",
        "Download Full Transaction Report"
    )

    st.divider()

    st.subheader("Account Report")

    download_csv(
        create_account_summary(results),
        "aml_account_report.csv",
        "Download Account Report"
    )

    st.divider()

    st.subheader("Network Report")

    download_csv(
        create_network_summary(results),
        "aml_network_report.csv",
        "Download Network Report"
    )

    st.divider()

    # Generate a compact summary of the current analysis.
    st.subheader("Analysis Summary")

    report_summary = {
        "Total Transactions": total_transactions,
        "Unique Sender Accounts": unique_senders,
        "Unique Receiver Accounts": unique_receivers,
        "Unique Banks": total_banks,
        "Trained Model Available": model_available,
        "Review Threshold": threshold,
    }

    if model_available:

        report_summary["Flagged Transactions"] = (
            flagged_transactions
        )

        report_summary["Flagged Percentage"] = round(
            flagged_percentage,
            2
        )

    st.json(report_summary)


# ============================================================
# 6. MODEL AND EVALUATION METRICS
# ============================================================

with metrics_tab:

    st.header("Model Information & Evaluation")

    if artifact is None:

        st.warning(
            "No trained model is currently available. "
            "Run train.py to generate aml_model.joblib."
        )

    else:

        st.success("Trained model successfully loaded.")

        st.write(
            "**Model type:** HistGradientBoostingClassifier"
        )

        st.write(
            f"**Default threshold:** {artifact.get('threshold', 0.5)}"
        )

        st.subheader("Model Features")

        # Show the feature inputs used by the transaction classifier.
        st.code(
            "\n".join(artifact["features"]),
            language="text"
        )

    st.divider()

    st.subheader("Training Evaluation Results")

    if metrics is None:

        st.info(
            "No evaluation_metrics.json file was found. "
            "Your current train.py saves the model but does not "
            "save a separate metrics JSON file."
        )

    else:

        # Display metrics only when they have actually been generated.
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        metric_col1.metric(
            "ROC-AUC",
            f"{metrics.get('roc_auc', 0):.4f}"
        )

        metric_col2.metric(
            "Average Precision",
            f"{metrics.get('average_precision', 0):.4f}"
        )

        metric_col3.metric(
            "F1 Score",
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

        # Display the classification report if it exists.
        if "classification_report" in metrics:

            st.subheader("Classification Report")

            classification_df = pd.DataFrame(
                metrics["classification_report"]
            ).transpose()

            st.dataframe(
                classification_df,
                use_container_width=True
            )

    st.caption(
        "Risk scores are model-generated indicators for human review. "
        "They are not proof of money laundering."
    )


# ============================================================
# APPLICATION FOOTER
# ============================================================

st.divider()

st.caption(
    "AML Statement Analyzer | Academic Prototype | "
    "Automated flags require human investigation."
)
