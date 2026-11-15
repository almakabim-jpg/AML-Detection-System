import pandas as pd
import numpy as np

REQUIRED = ['Timestamp','From Bank','Account','To Bank','Account.1','Amount Received','Receiving Currency','Amount Paid','Payment Currency','Payment Format']
FEATURES = ['amount_paid','amount_received','hour','day_of_week','same_account','same_bank','currency_mismatch','amount_difference','sender_prior_count','receiver_prior_count','sender_prior_total','receiver_prior_total','payment_format_code']

def prepare(df):
    df = df.copy()
    # Pandas renames the second Account column Account.1.
    if 'Account.1' not in df.columns and 'To Account' in df.columns:
        df = df.rename(columns={'To Account':'Account.1'})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing: raise ValueError('Missing columns: ' + ', '.join(missing))
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp']).sort_values('Timestamp', kind='stable').reset_index(drop=True)
    for col in ['Amount Paid','Amount Received']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)
    sender = df['From Bank'].astype(str) + ':' + df['Account'].astype(str)
    receiver = df['To Bank'].astype(str) + ':' + df['Account.1'].astype(str)
    df['amount_paid'] = np.log1p(df['Amount Paid'])
    df['amount_received'] = np.log1p(df['Amount Received'])
    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['same_account'] = (sender == receiver).astype(int)
    df['same_bank'] = (df['From Bank'].astype(str) == df['To Bank'].astype(str)).astype(int)
    df['currency_mismatch'] = (df['Receiving Currency'] != df['Payment Currency']).astype(int)
    # Different currencies are not directly comparable; use 0 in that case.
    df['amount_difference'] = np.where(df['currency_mismatch'] == 0, np.log1p(abs(df['Amount Paid']-df['Amount Received'])), 0)
    df['sender_prior_count'] = df.groupby(sender, sort=False).cumcount()
    df['receiver_prior_count'] = df.groupby(receiver, sort=False).cumcount()
    df['sender_prior_total'] = df.groupby(sender, sort=False)['Amount Paid'].cumsum() - df['Amount Paid']
    df['receiver_prior_total'] = df.groupby(receiver, sort=False)['Amount Received'].cumsum() - df['Amount Received']
    df['sender_prior_total'] = np.log1p(df['sender_prior_total'].clip(lower=0))
    df['receiver_prior_total'] = np.log1p(df['receiver_prior_total'].clip(lower=0))
    formats = {'ACH':1,'Cheque':2,'Credit Card':3,'Reinvestment':4,'Wire':5,'Cash':6,'Bitcoin':7}
    df['payment_format_code'] = df['Payment Format'].astype(str).map(formats).fillna(0).astype(int)
    return df
