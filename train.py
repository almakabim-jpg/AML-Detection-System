"""Train a baseline AML transaction classifier. Usage: python train.py path/to/HI-Small_Trans.csv"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
import joblib
from common import prepare, FEATURES

if len(sys.argv) < 2:
    raise SystemExit('Usage: python train.py "path/to/HI-Small_Trans.csv"')
path = Path(sys.argv[1])
if not path.exists(): raise SystemExit(f'File not found: {path}')
print('Reading dataset. This may take several minutes...')
df = pd.read_csv(path, dtype={'From Bank':str,'To Bank':str,'Account':str,'Account.1':str}, low_memory=False)
if 'Is Laundering' not in df: raise SystemExit('Missing Is Laundering label')
df['Is Laundering'] = pd.to_numeric(df['Is Laundering'], errors='coerce')
df = df[df['Is Laundering'].isin([0,1])]
print('Actual label counts:', df['Is Laundering'].value_counts().to_dict())
if df['Is Laundering'].nunique() < 2: raise SystemExit('Both classes are required. Check the complete CSV.')
print('Computing features...')
df = prepare(df)
# Chronological holdout: later transactions are reserved for evaluation.
cut = int(len(df)*0.8)
train, test = df.iloc[:cut], df.iloc[cut:]
if train['Is Laundering'].nunique()<2 or test['Is Laundering'].nunique()<2:
    raise SystemExit('Temporal split has only one class in train or test. Inspect label dates and adjust evaluation design.')
# Sample only the training negatives for tractability; keep every training positive.
pos = train[train['Is Laundering']==1]
neg = train[train['Is Laundering']==0].sample(n=min(250000, int((train['Is Laundering']==0).sum())), random_state=42)
train_sample = pd.concat([pos,neg]).sample(frac=1,random_state=42)
print(f'Training on {len(train_sample):,} rows; testing on {len(test):,} untouched later rows')
model = HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=20, learning_rate=0.08, random_state=42)
model.fit(train_sample[FEATURES], train_sample['Is Laundering'])
y = test['Is Laundering'].astype(int)
prob = model.predict_proba(test[FEATURES])[:,1]
# Negative undersampling changes the apparent class prior. Raw scores are not calibrated probabilities.
pred = (prob >= 0.5).astype(int)
print('Test prevalence:', float(y.mean()))
print('PR-AUC:', round(average_precision_score(y,prob),5))
print('ROC-AUC:', round(roc_auc_score(y,prob),5))
print('Confusion matrix:\n',confusion_matrix(y,pred))
print(classification_report(y,pred,zero_division=0))
output = Path(__file__).parent/'aml_model.joblib'
joblib.dump({'model':model,'features':FEATURES,'threshold':0.5},output)
print('Saved:',output)
print('Caution: score is an uncalibrated risk ranking, not a verified laundering probability.')
