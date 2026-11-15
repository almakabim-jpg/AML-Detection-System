import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from common import prepare, FEATURES

st.set_page_config(page_title='AML Statement Analyzer',layout='wide')
st.title('AML Statement Analyzer')
st.caption('Academic prototype • Synthetic-data model • Flags require human review')
model_path = Path(__file__).parent/'aml_model.joblib'
if not model_path.exists():
    st.warning('Model not found. Run train.py first to create aml_model.joblib.')
    st.stop()
artifact = joblib.load(model_path)
file = st.file_uploader('Upload a transaction-history CSV in IBM HI-Small column format',type='csv')
st.info('This prototype accepts structured CSV transaction histories, not PDF bank statements. It does not make legal determinations.')
if file is not None:
    try:
        raw = pd.read_csv(file, dtype={'From Bank':str,'To Bank':str,'Account':str,'Account.1':str})
        if raw.empty: raise ValueError('CSV is empty')
        clean = prepare(raw)
        score = artifact['model'].predict_proba(clean[FEATURES])[:,1]
        clean['Risk score (uncalibrated)'] = score
        threshold = st.slider('Review threshold',0.0,1.0,float(artifact['threshold']),0.05)
        clean['Flag for review'] = clean['Risk score (uncalibrated)'] >= threshold
        a,b,c = st.columns(3)
        a.metric('Transactions',f'{len(clean):,}')
        b.metric('Flagged for review',f'{int(clean["Flag for review"].sum()):,}')
        c.metric('Flagged percentage',f'{100*clean["Flag for review"].mean():.2f}%')
        st.subheader('Highest-risk transactions')
        show = ['Timestamp','From Bank','Account','To Bank','Account.1','Amount Paid','Payment Currency','Payment Format','Risk score (uncalibrated)','Flag for review']
        st.dataframe(clean.sort_values('Risk score (uncalibrated)',ascending=False)[show].head(100),use_container_width=True)
        st.subheader('Account connections')
        links = clean.groupby(['From Bank','Account','To Bank','Account.1'],dropna=False).agg(transactions=('Amount Paid','size'),total_paid=('Amount Paid','sum')).reset_index()
        st.dataframe(links.sort_values('transactions',ascending=False).head(50),use_container_width=True)
        st.download_button('Download analyzed transactions',clean.to_csv(index=False).encode('utf-8'),'aml_review_results.csv','text/csv')
        st.caption('Scores are computed from only the uploaded history. Incomplete statements can miss connected accounts and prior activity.')
    except Exception as exc:
        st.error(f'Unable to analyze file: {exc}')
