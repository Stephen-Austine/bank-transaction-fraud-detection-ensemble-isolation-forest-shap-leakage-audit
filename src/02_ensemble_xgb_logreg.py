"""
v2 - Leakage-free ensemble: XGBoost + Logistic Regression (soft voting), ADASYN
oversampling on the training split, Isolation Forest anomaly flag and a hybrid score.
Risk_Score is deliberately NOT used as a feature.
"""
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import IsolationForest, VotingClassifier
from imblearn.over_sampling import ADASYN

import xgboost as xgb
import lightgbm as lgb

np.random.seed(42)

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "complete_dirty_bank_transactions.csv"
DATA_CLEAN = ROOT / "data" / "processed" / "cleaned_transactions.csv"
OUT = ROOT / "outputs"
FIG = ROOT / "assets" / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


# Load and clean data
df = pd.read_csv(DATA_RAW)

money_cols = ['Loan_Amount', 'Account_Balance', 'Recent_Transaction_Amount', 'Avg_Monthly_Transaction']
for col in money_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(r'[^\d\.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

df[money_cols] = df[money_cols].fillna(df[money_cols].median())
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Credit_Score'] = df['Credit_Score'].fillna(df['Credit_Score'].median())
df['Transaction_Time'] = pd.to_datetime(df['Transaction_Time'], errors='coerce')
df = df.dropna(subset=['Transaction_Time'])

# Feature Engineering
df['Hour'] = df['Transaction_Time'].dt.hour
df['Weekday'] = df['Transaction_Time'].dt.dayofweek
df['Trans_to_Balance_Ratio'] = df['Recent_Transaction_Amount'] / (df['Account_Balance'] + 1)
df['Trans_Amount_Deviation'] = abs(df['Recent_Transaction_Amount'] - df['Avg_Monthly_Transaction']) / (df['Avg_Monthly_Transaction'] + 1)
df['Trans_Frequency'] = df.groupby('Customer_ID')['Transaction_Time'].transform('count')

df = df.sort_values(['Customer_ID', 'Transaction_Time'])
df['Time_Since_Last_Trans'] = df.groupby('Customer_ID')['Transaction_Time'].diff().dt.total_seconds() / 3600
df['Time_Since_Last_Trans'] = df['Time_Since_Last_Trans'].fillna(df['Time_Since_Last_Trans'].median())
df['High_Velocity_Flag'] = (df['Time_Since_Last_Trans'] < 0.5).astype(int)

df['Device_Count'] = df.groupby('Customer_ID')['Device_Type'].transform('nunique')
df['Device_Inconsistency_Flag'] = (df['Device_Count'] > 1).astype(int)

df['Location_Inconsistency'] = (df['Transaction_Location'].str.split(',').str[1].str.strip() != df['State']).astype(int)

customer_trans_stats = df.groupby('Customer_ID')['Recent_Transaction_Amount'].agg(['mean', 'std']).reset_index()
customer_trans_stats.columns = ['Customer_ID', 'Trans_Amount_Mean', 'Trans_Amount_Std']
df = df.merge(customer_trans_stats, on='Customer_ID')
df['Unusual_Amount_Flag'] = (
    (df['Recent_Transaction_Amount'] < (df['Trans_Amount_Mean'] - 2.5 * df['Trans_Amount_Std'])) |
    (df['Recent_Transaction_Amount'] > (df['Trans_Amount_Mean'] + 2.5 * df['Trans_Amount_Std']))
).astype(int).fillna(0)

df['Trans_Amount_Rank'] = df.groupby('Customer_ID')['Recent_Transaction_Amount'].rank(method='dense', ascending=False)

# Encode categorical
categorical_cols = ['Gender', 'Account_Type', 'Transaction_Type', 'Pay_On_Time', 'Transaction_Location', 'Device_Type', 'Business_Type']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Define features
features = [
    'Age', 'Gender', 'Account_Balance', 'Credit_Score', 'Loan_Amount', 'Transaction_Type',
    'Pay_On_Time', 'Recent_Transaction_Amount', 'Avg_Monthly_Transaction', 'Device_Type',
    'Account_Age_Years', 'Hour', 'Weekday', 'Trans_to_Balance_Ratio', 
    'Trans_Amount_Deviation', 'Trans_Frequency', 'Time_Since_Last_Trans', 
    'High_Velocity_Flag', 'Device_Inconsistency_Flag', 'Location_Inconsistency', 
    'Unusual_Amount_Flag', 'Trans_Amount_Rank'
]

X = df[features]
y = df['Is_Fraudulent']
X_scaled = StandardScaler().fit_transform(X)

adasyn = ADASYN(random_state=42, n_neighbors=2)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)

# === MODEL DEFINITIONS ===
xgb_model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.05,
    reg_lambda=2.0,
    reg_alpha=1.0,
    scale_pos_weight=100,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train_res, y_train_res)  # ✅ FIXED HERE

lr_model = LogisticRegression(penalty='l2', C=0.05, class_weight='balanced', random_state=42)

ensemble_model = VotingClassifier(
    estimators=[('xgb', xgb_model), ('lr', lr_model)],
    voting='soft'
)
ensemble_model.fit(X_train_res, y_train_res)

# === CV and Feature Importance ===
cv_scores = cross_val_score(ensemble_model, X_scaled, y, cv=5, scoring='recall_macro')
print("Cross-Validation Recall-Macro Scores:", cv_scores)
print("Mean CV Recall-Macro:", cv_scores.mean(), "±", cv_scores.std())

xgb_importance = pd.DataFrame({
    'Feature': features,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nXGBoost Feature Importance:\n", xgb_importance)

# === SHAP EXPLAINABILITY ===
explainer = shap.Explainer(xgb_model, X_scaled)
shap_values = explainer(X_scaled[:100])
shap.summary_plot(shap_values, features=X.iloc[:100], feature_names=features, show=False)
plt.tight_layout()
plt.savefig(FIG / "v2_shap_summary.png", dpi=150)

# === UNSUPERVISED ISOLATION FOREST ===
iso_forest = IsolationForest(contamination=0.08, random_state=42)
df['Anomaly_Score'] = iso_forest.fit(X_scaled).score_samples(X_scaled)
df['Is_Anomaly'] = (iso_forest.predict(X_scaled) == -1).astype(int)

# === PREDICT AND EVALUATE ===
df['Ensemble_Prediction'] = ensemble_model.predict_proba(X_scaled)[:, 1]
df['Hybrid_Fraud_Score'] = (df['Ensemble_Prediction'] + df['Is_Anomaly']) / 2
df['Hybrid_Fraud_Prediction'] = (df['Hybrid_Fraud_Score'] >= 0.4).astype(int)

y_pred = (ensemble_model.predict_proba(X_test)[:, 1] >= 0.4).astype(int)
y_proba = ensemble_model.predict_proba(X_test)[:, 1]

print("\nSupervised Ensemble Model Results:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))

missed_frauds = df[(df['Is_Fraudulent'] == 0) & (df['Hybrid_Fraud_Prediction'] == 1)]
print(f"\nNumber of potentially missed frauds detected: {len(missed_frauds)}")
missed_frauds.to_csv(OUT / 'missed_frauds_v2_xgb_logreg.csv', index=False)

# === PLOT FEATURE IMPORTANCE ===
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=xgb_importance)
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig(FIG / 'v2_feature_importance.png', dpi=150)
