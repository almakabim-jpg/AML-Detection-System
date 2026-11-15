"""Dark Streamlit interface for the AML Statement Analyzer.

The interface can be explored before model training. Before a trained model is
available, uploaded data can be profiled (counts, volumes and relationships),
but no transaction is labelled suspicious and no risk score is fabricated.
"""
from pathlib import Path
import json

import pandas as pd
import streamlit as st

try:
    import joblib
except ImportError:  # Allows the interface to render even before dependencies install.
    joblib = None

from common import FEATURES, prepare

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "aml_model.joblib"
METRICS_PATH = BASE_DIR / "evaluation_metrics.json"

st.set_page_config(
    page_title="AML Statement Analyzer",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------ Theme --------------------------------------
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #0b1120; color: #e5eaf3; }
[data-testid="stHeader"] { background: rgba(11,17,32,.92); }
[data-testid="stSidebar"] { background: #111b2e; border-right: 1px solid #26344d; }
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1500px; }
[data-testid="stMetric"] { background: linear-gradient(145deg,#17243a,#121d30); border: 1px solid #293b58; padding: 17px 18px; border-radius: 13px; }
[data-testid="stMetricLabel"] { color: #9eacc3; }
[data-testid="stMetricValue"] { color: #f4f7fb; }
section[data-testid="stFileUploaderDropzone"] { background: #121e32; border: 1px dashed #496080; border-radius: 12px; }
.stButton > button, .stDownloadButton > button { border-radius: 9px; font-weight: 650; min-height: 2.7rem; }
div[data-testid="stTabs"] button { color: #b8c5d8; }
.panel { background: #121e32; border: 1px solid #293b58; padding: 17px 19px; border-radius: 13px; }
.small-muted { color: #9eacc3; font-size: .88rem; }
.status-pill { display:inline-block; padding:5px 10px; border-radius:20px; background:#25354e; color:#c8d7ee; font-size:.78rem; }
hr { border-color: #26344d; }
</style>
""", unsafe_allow_html=True)

# ------------------------------ Helpers ------------------------------------
def read_csv_safely(file) -> pd.DataFrame:
    file.seek(0)
    df = pd.read_csv(
        file,
        dtype={"From Bank": str, "To Bank": str, "Account": str, "Account.1": str},
        low_memory=False,
    )
    if "Account.1" not in df.columns and "To Account" in df.columns:
        df = df.rename(columns={"To Account": "Account.1"})
    return df


def money_label(value) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def account_summary(df: pd.DataFrame) -> pd.DataFrame:
    sender = df.groupby(["From Bank", "Account"], dropna=False).agg(
        transactions_sent=("Amount Paid", "size"),
        total_sent=("Amount Paid", "sum"),
        unique_recipients=("Account.1", "nunique"),
    ).reset_index().rename(columns={"From Bank": "Bank", "Account": "Account ID"})
    receiver = df.groupby(["To Bank", "Account.1"], dropna=False).agg(
        transactions_received=("Amount Received", "size"),
        total_received=("Amount Received", "sum"),
        unique_senders=("Account", "nunique"),
    ).reset_index().rename(columns={"To Bank": "Bank", "Account.1": "Account ID"})
    result = sender.merge(receiver, on=["Bank", "Account ID"], how="outer").fillna(0)
    for c in ["transactions_sent", "transactions_received", "unique_recipients", "unique_senders"]:
        result[c] = result[c].astype(int)
    result["total_flow"] = result["total_sent"] + result["total_received"]
    return result.sort_values("total_flow", ascending=False)


def network_summary(df: pd.DataFrame) -> pd.DataFrame:
    result = df.groupby(
        ["From Bank", "Account", "To Bank", "Account.1"], dropna=False
    ).agg(
        transaction_count=("Amount Paid", "size"),
        total_amount=("Amount Paid", "sum"),
        first_seen=("Timestamp", "min"),
        last_seen=("Timestamp", "max"),
    ).reset_index()
    return result.sort_values(["transaction_count", "total_amount"], ascending=False)


def load_model():
    if not MODEL_PATH.exists() or joblib is None:
        return None, None
    try:
        artifact = joblib.load(MODEL_PATH)
        return artifact, None
    except Exception as exc:
        return None, str(exc)


# ------------------------------ Sidebar ------------------------------------
artifact, model_error = load_model()
model_ready = artifact is not None
with st.sidebar:
    st.markdown("## 🔎 AML ANALYZER")
    st.caption("Financial transaction risk review")
    st.markdown("---")
    st.markdown("**SYSTEM STATUS**")
    if model_ready:
        st.success("Trained model connected")
    elif model_error:
        st.error("Model file could not be loaded")
    else:
        st.warning("Demo / pre-training mode")
    st.markdown("---")
    st.markdown("**INPUT REQUIREMENTS**")
    st.write("• CSV transaction history")
    st.write("• IBM HI-Small-style columns")
    st.caption("PDF and image statements are not supported in this prototype.")
    st.markdown("---")
    st.markdown("**IMPORTANT**")
    st.caption("Flags are review indicators, not proof of money laundering. A human must investigate.")

# ------------------------------ Header -------------------------------------
left, right = st.columns([4, 1])
with left:
    st.markdown("# AML Statement Analyzer")
    st.markdown("Identify transaction patterns and prioritize activity for human review.")
with right:
    st.markdown('<div style="text-align:right;padding-top:15px"><span class="status-pill">ACADEMIC PROTOTYPE</span></div>', unsafe_allow_html=True)

# ------------------------------ Overview cards -----------------------------
if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None
if "scored_df" not in st.session_state:
    st.session_state.scored_df = None

current = st.session_state.analysis_df
if current is not None and not current.empty:
    total_tx = len(current)
    total_accounts = pd.concat([current["Account"].astype(str), current["Account.1"].astype(str)]).nunique()
    total_volume = pd.to_numeric(current["Amount Paid"], errors="coerce").fillna(0).sum()
else:
    total_tx, total_accounts, total_volume = 0, 0, 0
flagged_count = int(st.session_state.scored_df["Flag for review"].sum()) if st.session_state.scored_df is not None else None
m1, m2, m3, m4 = st.columns(4)
m1.metric("Transactions loaded", f"{total_tx:,}" if current is not None else "—")
m2.metric("Accounts observed", f"{total_accounts:,}" if current is not None else "—")
m3.metric("Total transaction volume", money_label(total_volume) if current is not None else "—")
m4.metric("Flagged for review", f"{flagged_count:,}" if flagged_count is not None else "Unavailable")

st.markdown("---")

# ------------------------------ Main tabs ----------------------------------
tab_dashboard, tab_analyze, tab_transactions, tab_accounts, tab_network, tab_reports, tab_model = st.tabs([
    "◈ Dashboard", "＋ New Analysis", "▤ Transactions", "◎ Accounts", "⌘ Network", "⇩ Reports", "⚙ Model & Metrics"
])

with tab_dashboard:
    st.subheader("Dashboard overview")
    if current is None:
        st.info("Upload a transaction-history CSV in **New Analysis** to populate the dashboard.")
        st.markdown("<div class='panel'><b>Getting started</b><p class='small-muted'>1. Upload a CSV file · 2. Choose an analysis type · 3. Run the analysis · 4. Review the results and export them.</p></div>", unsafe_allow_html=True)
    else:
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Transaction volume by payment currency**")
            currency_volume = current.groupby("Payment Currency")["Amount Paid"].sum().sort_values(ascending=False).head(10)
            st.bar_chart(currency_volume)
        with d2:
            st.markdown("**Transactions over time**")
            dates = pd.to_datetime(current["Timestamp"], errors="coerce")
            daily = dates.dt.date.value_counts().sort_index()
            daily.index = daily.index.astype(str)
            st.line_chart(daily)
        if flagged_count is None:
            st.info("No model predictions are available yet. Charts above describe the uploaded data only; they do not identify laundering.")
        else:
            st.success(f"The connected model flagged {flagged_count:,} transaction(s) for human review at the selected threshold.")

with tab_analyze:
    st.subheader("Start a new analysis")
    st.write("Upload structured transaction history and choose the scope of analysis.")
    uploaded = st.file_uploader("Upload transaction history", type=["csv"], key="aml_csv", help="CSV only. PDF and image statements are not supported.")
    c1, c2 = st.columns([2, 1])
    with c1:
        analysis_type = st.selectbox(
            "Analysis type",
            ["Comprehensive analysis (all three)", "Transaction-level analysis", "Account-level analysis", "Network-level analysis"],
            help="Transaction scoring requires the trained model. Account and network summaries describe relationships in the uploaded file.",
        )
    with c2:
        threshold = st.slider("Review threshold", 0.05, 0.95, 0.50, 0.05, help="Transactions at or above this model score are flagged. Only used when a trained model is available.")
    if not model_ready:
        st.warning("Pre-training mode: you can inspect the uploaded data and account/network activity. Risk scoring and AML flags will remain unavailable until a trained model is installed.")
    if st.button("Analyze Transactions", type="primary", use_container_width=True, disabled=(uploaded is None)):
        try:
            with st.spinner("Loading and profiling transaction history..."):
                raw = read_csv_safely(uploaded)
                required = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format"]
                missing = [c for c in required if c not in raw.columns]
                if missing:
                    raise ValueError("Missing required CSV columns: " + ", ".join(missing))
                raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], errors="coerce")
                raw = raw.dropna(subset=["Timestamp"]).copy()
                for col in ["Amount Paid", "Amount Received"]:
                    raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0).clip(lower=0)
                if raw.empty:
                    raise ValueError("No rows with valid timestamps were found.")
                st.session_state.analysis_df = raw.reset_index(drop=True)
                st.session_state.scored_df = None
                st.session_state.analysis_type = analysis_type
                st.session_state.threshold = threshold
                if model_ready:
                    prepared = prepare(raw)
                    if list(artifact.get("features", FEATURES)) != list(FEATURES):
                        raise ValueError("The model feature definitions do not match common.py. Please retrain the model.")
                    scores = artifact["model"].predict_proba(prepared[FEATURES])[:, 1]
                    prepared["Risk score (uncalibrated)"] = scores
                    prepared["Flag for review"] = scores >= threshold
                    st.session_state.scored_df = prepared
            st.success(f"Successfully loaded and profiled {len(raw):,} transaction(s).")
            if not model_ready:
                st.info("Data profiling is complete. No risk scores or suspicious-transaction labels were generated because the model has not been trained.")
            else:
                st.success("Model scoring is complete. Review flagged records in the Transactions tab.")
        except Exception as exc:
            st.error(f"Analysis could not be completed: {exc}")
    if current is not None:
        st.caption(f"Last loaded analysis: {len(current):,} transaction rows. Open the other tabs to explore the uploaded data.")

with tab_transactions:
    st.subheader("Transaction investigation")
    scored = st.session_state.scored_df
    if scored is None:
        st.info("Transaction-level risk scores will appear here after a trained model is connected and the uploaded file is analyzed.")
        if current is not None:
            st.markdown("**Uploaded transaction preview**")
            st.dataframe(current.head(100), use_container_width=True, hide_index=True)
    else:
        show_flagged = st.checkbox("Show flagged transactions only", value=False)
        view = scored[scored["Flag for review"]] if show_flagged else scored
        st.caption(f"Showing {len(view):,} of {len(scored):,} transactions")
        st.dataframe(view.sort_values("Risk score (uncalibrated)", ascending=False), use_container_width=True, hide_index=True)

with tab_accounts:
    st.subheader("Account investigation")
    if current is None:
        st.info("Upload and analyze a CSV to view account activity summaries.")
    else:
        accounts = account_summary(current)
        a1, a2 = st.columns(2)
        a1.metric("Unique accounts", f"{len(accounts):,}")
        a2.metric("Accounts with multiple senders", f"{int((accounts['unique_senders'] > 1).sum()):,}")
        st.caption("These are descriptive activity indicators, not model-generated suspicion labels.")
        st.dataframe(accounts.head(500), use_container_width=True, hide_index=True)

with tab_network:
    st.subheader("Network and transaction relationships")
    if current is None:
        st.info("Upload and analyze a CSV to inspect sender-recipient relationships.")
    else:
        network = network_summary(current)
        n1, n2, n3 = st.columns(3)
        n1.metric("Distinct sender-recipient links", f"{len(network):,}")
        n2.metric("Repeated links", f"{int((network['transaction_count'] > 1).sum()):,}")
        n3.metric("Links with 5+ transfers", f"{int((network['transaction_count'] >= 5).sum()):,}")
        st.caption("Relationship summaries help explore flow patterns. They do not establish that an account or transfer is involved in money laundering.")
        st.dataframe(network.head(500), use_container_width=True, hide_index=True)

with tab_reports:
    st.subheader("Reports and exports")
    if current is None:
        st.info("Run an analysis to make downloadable reports available.")
    else:
        st.download_button("Download uploaded transaction data (CSV)", current.to_csv(index=False).encode("utf-8"), "aml_transaction_profile.csv", "text/csv", use_container_width=True)
        if st.session_state.scored_df is not None:
            result = st.session_state.scored_df
            st.download_button("Download model results (CSV)", result.to_csv(index=False).encode("utf-8"), "aml_model_review_results.csv", "text/csv", use_container_width=True)
        else:
            st.caption("Model-results export will become available after model training and scoring.")

with tab_model:
    st.subheader("Model status and evaluation")
    if model_ready:
        st.success("A model artifact is available.")
        if METRICS_PATH.exists():
            try:
                metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
                metric_cols = st.columns(min(4, max(1, len(metrics))))
                for col, (name, value) in zip(metric_cols, metrics.items()):
                    if isinstance(value, (int, float)):
                        col.metric(name.replace("_", " ").title(), f"{value:.4f}")
                st.json(metrics)
            except Exception as exc:
                st.warning(f"Could not read evaluation_metrics.json: {exc}")
        else:
            st.info("Evaluation metrics file has not been added yet.")
    else:
        st.warning("No trained model is connected yet.")
        st.write("The interface and data-profiling features are available now. Transaction risk scoring will be enabled after training produces aml_model.joblib.")
        st.markdown("**Planned model evaluation**")
        st.write("Precision · Recall · F1-score · ROC-AUC · Average precision · Confusion matrix")

st.markdown("---")
st.caption("AML Statement Analyzer · Academic prototype · Analytical indicators require human review and are not legal determinations.")
