"""
app.py — Streamlit interface for the AML Statement Analyzer.

The interface is visible before training. The Analyze button stays disabled
until both a CSV is uploaded and aml_model.joblib is available.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from common import FEATURES, prepare

# Configure the page before rendering Streamlit elements.
st.set_page_config(
    page_title="AML Statement Analyzer",
    page_icon="🔎",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "aml_model.joblib"
model_ready = MODEL_PATH.exists()

# ---------------------------------------------------------------------
# 1. Header and sidebar
# ---------------------------------------------------------------------
st.title("AML Statement Analyzer")
st.caption(
    "Academic prototype | IBM AML transaction data | "
    "Flags require human review"
)

with st.sidebar:
    st.header("Project information")
    st.write(
        "Upload structured transaction histories to rank transactions "
        "for further review."
    )
    st.divider()
    st.subheader("Model status")
    if model_ready:
        st.success("Trained model available")
    else:
        st.warning("Model not trained yet")
    st.caption("The interface is available before model training.")
    st.divider()
    st.subheader("Accepted input")
    st.write("CSV transaction history")
    st.write("IBM HI-Small-style columns")

# ---------------------------------------------------------------------
# 2. File upload panel
# ---------------------------------------------------------------------
st.subheader("1. Upload transaction history")
uploaded_file = st.file_uploader(
    "Choose a transaction-history CSV file",
    type=["csv"],
    help="Upload a CSV with IBM HI-Small transaction columns.",
)
st.info(
    "This prototype accepts structured CSV transaction histories, not PDF "
    "or image-based bank statements. It does not make legal determinations."
)

# ---------------------------------------------------------------------
# 3. Analysis settings — visible even if no model exists yet
# ---------------------------------------------------------------------
st.subheader("2. Analysis settings")
left, right = st.columns(2)

with left:
    st.selectbox(
        "Analysis type",
        ["Transaction-level risk review"],
        help="The current baseline model scores individual transactions.",
    )

default_threshold = 0.50
if model_ready:
    try:
        default_threshold = float(
            joblib.load(MODEL_PATH).get("threshold", 0.50)
        )
    except Exception:
        st.warning("The model file exists but could not be read.")
with right:
    threshold = st.slider(
        "Review threshold", 0.0, 1.0, default_threshold, 0.05,
        help="Scores at or above this value are flagged for human review.",
    )

if not model_ready:
    st.warning(
        "The interface is ready, but the model has not been trained. "
        "Run train.py with the labelled HI-Small_Trans.csv file, then make "
        "aml_model.joblib available beside app.py."
    )

# ---------------------------------------------------------------------
# 4. Run analysis only when file and model are both available
# ---------------------------------------------------------------------
analyze = st.button(
    "Analyze transactions",
    type="primary",
    use_container_width=True,
    disabled=(uploaded_file is None or not model_ready),
)

if analyze and uploaded_file is not None:
    try:
        with st.spinner("Preprocessing and scoring transactions..."):
            artifact = joblib.load(MODEL_PATH)
            raw = pd.read_csv(
                uploaded_file,
                dtype={
                    "From Bank": str, "To Bank": str,
                    "Account": str, "Account.1": str,
                },
                low_memory=False,
            )
            clean = prepare(raw)

            # Ensure the model artifact and current preprocessing agree.
            if list(artifact.get("features", FEATURES)) != list(FEATURES):
                raise ValueError(
                    "Model features do not match common.py. Retrain the model."
                )

            scores = artifact["model"].predict_proba(clean[FEATURES])[:, 1]
            clean["Risk score (uncalibrated)"] = scores
            clean["Flag for review"] = scores >= threshold

        # -----------------------------------------------------------------
        # 5. Summary metric cards
        # -----------------------------------------------------------------
        st.divider()
        st.subheader("3. Analysis summary")
        total = len(clean)
        flagged = int(clean["Flag for review"].sum())
        percentage = 100 * flagged / total if total else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Transactions analyzed", f"{total:,}")
        col2.metric("Flagged for review", f"{flagged:,}")
        col3.metric("Flagged percentage", f"{percentage:.2f}%")

        # -----------------------------------------------------------------
        # 6. Risk-score chart
        # -----------------------------------------------------------------
        st.subheader("4. Risk-score distribution")
        bins = pd.cut(
            clean["Risk score (uncalibrated)"],
            bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.001],
            labels=["0.00–0.20", "0.20–0.40", "0.40–0.60",
                    "0.60–0.80", "0.80–1.00"],
            include_lowest=True,
        )
        distribution = bins.value_counts(sort=False).rename("Transactions")
        st.bar_chart(distribution)

        # -----------------------------------------------------------------
        # 7. Highest-risk transaction table
        # -----------------------------------------------------------------
        st.subheader("5. Highest-risk transactions")
        columns = [
            "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
            "Amount Paid", "Payment Currency", "Payment Format",
            "Risk score (uncalibrated)", "Flag for review",
        ]
        highest_risk = clean.sort_values(
            "Risk score (uncalibrated)", ascending=False
        )[columns].head(100)
        st.dataframe(highest_risk, use_container_width=True, hide_index=True)

        # -----------------------------------------------------------------
        # 8. Account-connection summary
        # -----------------------------------------------------------------
        st.subheader("6. Account connections")
        connections = (
            clean.groupby(
                ["From Bank", "Account", "To Bank", "Account.1"],
                dropna=False,
            )
            .agg(
                transactions=("Amount Paid", "size"),
                total_amount_paid=("Amount Paid", "sum"),
            )
            .reset_index()
            .sort_values("transactions", ascending=False)
            .head(50)
        )
        st.dataframe(connections, use_container_width=True, hide_index=True)

        # -----------------------------------------------------------------
        # 9. Download results for further review
        # -----------------------------------------------------------------
        st.subheader("7. Export results")
        st.download_button(
            "Download analyzed transactions (CSV)",
            data=clean.to_csv(index=False).encode("utf-8"),
            file_name="aml_review_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(
            "Scores are uncalibrated risk rankings, not verified probabilities. "
            "A flag is not proof of money laundering; human review is required."
        )

    except Exception as error:
        st.error(f"Unable to analyze the uploaded file: {error}")

# ---------------------------------------------------------------------
# 10. Responsible-use footer
# ---------------------------------------------------------------------
st.divider()
st.caption(
    "Academic research prototype. Results depend on the uploaded data, "
    "preprocessing, training sample, and review threshold."
)
