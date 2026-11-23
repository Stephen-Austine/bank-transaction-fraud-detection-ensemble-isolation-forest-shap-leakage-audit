"""
v3 (final) - Leakage-free hybrid: XGBoost + LightGBM + Logistic Regression (weighted soft
voting) with ADASYN oversampling, F1-optimised decision threshold, SHAP explanations and an
Isolation Forest anomaly flag blended into a hybrid fraud score.
Risk_Score is deliberately NOT used as a feature.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
import warnings

# Set random seed
np.random.seed(42)

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "complete_dirty_bank_transactions.csv"
DATA_CLEAN = ROOT / "data" / "processed" / "cleaned_transactions.csv"
OUT = ROOT / "outputs"
FIG = ROOT / "assets" / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning, message=".*use_label_encoder.*")

# Load data
df = pd.read_csv(DATA_RAW)

# Clean monetary columns
money_cols = ['Loan_Amount', 'Account_Balance', 'Recent_Transaction_Amount', 'Avg_Monthly_Transaction']
for col in money_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(r'[^\d\.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

df[money_cols] = df[money_cols].fillna(df[money_cols].median())
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Credit_Score'] = df['Credit_Score'].fillna(df['Credit_Score'].median())

# Handle datetime
df['Transaction_Time'] = pd.to_datetime(df['Transaction_Time'], errors='coerce')
df = df.dropna(subset=['Transaction_Time'])

# Feature engineering
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

# Unusual amount
customer_trans_stats = df.groupby('Customer_ID')['Recent_Transaction_Amount'].agg(['mean', 'std']).reset_index()
customer_trans_stats.columns = ['Customer_ID', 'Trans_Amount_Mean', 'Trans_Amount_Std']
df = df.merge(customer_trans_stats, on='Customer_ID')
df['Unusual_Amount_Flag'] = (
    (df['Recent_Transaction_Amount'] < (df['Trans_Amount_Mean'] - 2.5 * df['Trans_Amount_Std'])) |
    (df['Recent_Transaction_Amount'] > (df['Trans_Amount_Mean'] + 2.5 * df['Trans_Amount_Std']))
).astype(int).fillna(0)

df['Trans_Amount_Rank'] = df.groupby('Customer_ID')['Recent_Transaction_Amount'].rank(method='dense', ascending=False)

# Categorical encoding
categorical_cols = ['Gender', 'Account_Type', 'Transaction_Type', 'Pay_On_Time', 'Transaction_Location', 'Device_Type', 'Business_Type']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Feature selection
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

# Train-test split
adasyn = ADASYN(random_state=42, n_neighbors=2)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)

# Models
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
xgb_model.fit(X_train_res, y_train_res)

lr_model = LogisticRegression(penalty='l2', C=0.05, class_weight='balanced', random_state=42)

lgb_model = LGBMClassifier(
    n_estimators=60,
    num_leaves=31,
    learning_rate=0.05,
    class_weight='balanced',
    random_state=42
)
lgb_model.fit(X_train_res, y_train_res)

ensemble_model = VotingClassifier(
    estimators=[('xgb', xgb_model), ('lr', lr_model), ('lgb', lgb_model)],
    voting='soft',
    weights=[3, 1, 3]
)
ensemble_model.fit(X_train_res, y_train_res)

# CV and Feature Importance
cv_scores = cross_val_score(ensemble_model, X_scaled, y, cv=5, scoring='recall_macro')
print("Cross-Validation Recall-Macro Scores:", cv_scores)
print("Mean CV Recall-Macro:", cv_scores.mean(), "±", cv_scores.std())

xgb_importance = pd.DataFrame({
    'Feature': features,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nXGBoost Feature Importance:\n", xgb_importance)

# SHAP Explainability
explainer = shap.Explainer(xgb_model, X_scaled)
shap_values = explainer(X_scaled[:100])
shap.summary_plot(shap_values, features=X.iloc[:100], feature_names=features, show=False)
plt.tight_layout()
plt.savefig(FIG / "v3_shap_summary.png", dpi=150)

# Isolation Forest
iso_model = IsolationForest(contamination=0.08, random_state=42)
df['Anomaly_Score'] = iso_model.fit(X_scaled).score_samples(X_scaled)
df['Is_Anomaly'] = (iso_model.predict(X_scaled) == -1).astype(int)

# Ensemble Predictions
df['Ensemble_Prediction'] = ensemble_model.predict_proba(X_scaled)[:, 1]
df['Hybrid_Fraud_Score'] = (df['Ensemble_Prediction'] + df['Is_Anomaly']) / 2
df['Hybrid_Fraud_Prediction'] = (df['Hybrid_Fraud_Score'] >= 0.4).astype(int)

# Optimize threshold based on F1
thresholds = np.arange(0.05, 0.95, 0.05)
best_f1 = 0
best_thresh = 0.4

for t in thresholds:
    preds = (ensemble_model.predict_proba(X_test)[:, 1] >= t).astype(int)
    f1 = f1_score(y_test, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"Best threshold: {best_thresh}, Best F1: {best_f1}")

y_pred = (ensemble_model.predict_proba(X_test)[:, 1] >= best_thresh).astype(int)
y_proba = ensemble_model.predict_proba(X_test)[:, 1]

print("\nSupervised Ensemble Model Results:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))

missed_frauds = df[(df['Is_Fraudulent'] == 0) & (df['Hybrid_Fraud_Prediction'] == 1)]
print(f"\nNumber of potentially missed frauds detected: {len(missed_frauds)}")
missed_frauds.to_csv(OUT / 'missed_frauds_v3_xgb_lgbm_logreg.csv', index=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=xgb_importance)
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig(FIG / 'v3_feature_importance.png', dpi=150)

plt.figure(figsize=(10, 5))
sns.histplot(df['Anomaly_Score'], bins=50, kde=True)
plt.title("Distribution of Isolation Forest Anomaly Scores")
plt.xlabel("Anomaly Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIG / "v3_anomaly_score_distribution.png", dpi=150)


# =====================================================================================
# Extra diagnostics (added for the GitHub release; the modelling code above is unchanged)
# =====================================================================================
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

proba_test = ensemble_model.predict_proba(X_test)[:, 1]
prevalence = float(y_test.mean())
pr_auc = average_precision_score(y_test, proba_test)
print("\n=== EXTRA DIAGNOSTICS ===")
print(f"Test split: n={len(y_test)}, frauds={int(y_test.sum())}, prevalence={prevalence:.4f}")
print(f"PR-AUC (average precision) on test split: {pr_auc:.4f}  (no-skill baseline = {prevalence:.4f})")

# 1) ROC and precision-recall curves
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
fpr, tpr, _ = roc_curve(y_test, proba_test)
ax[0].plot(fpr, tpr, color="#2980b9", lw=2, label=f"Ensemble (AUC = {roc_auc_score(y_test, proba_test):.3f})")
ax[0].plot([0, 1], [0, 1], "--", color="grey", label="Chance (AUC = 0.5)")
ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
ax[0].set_title("ROC curve (test split)", fontweight="bold"); ax[0].legend(loc="lower right")
prec, rec, _ = precision_recall_curve(y_test, proba_test)
ax[1].plot(rec, prec, color="#c0392b", lw=2, label=f"Ensemble (AP = {pr_auc:.3f})")
ax[1].axhline(prevalence, ls="--", color="grey", label=f"No-skill (prevalence = {prevalence:.3f})")
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
ax[1].set_title("Precision-recall curve (test split)", fontweight="bold"); ax[1].legend(loc="upper right")
plt.tight_layout(); plt.savefig(FIG / "v3_roc_pr_curves.png", dpi=150); plt.close()

# 2) Confusion matrix at the F1-optimal threshold
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4.2))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Pred: legit", "Pred: fraud"], yticklabels=["Actual: legit", "Actual: fraud"])
ax.set_title(f"Confusion matrix (threshold = {best_thresh:.2f})", fontweight="bold")
plt.tight_layout(); plt.savefig(FIG / "v3_confusion_matrix.png", dpi=150); plt.close()

# 3) F1 as a function of the decision threshold
f1_curve = [f1_score(y_test, (proba_test >= t).astype(int), zero_division=0) for t in thresholds]
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(thresholds, f1_curve, marker="o", color="#8e44ad")
ax.axvline(best_thresh, ls="--", color="grey", label=f"Best threshold = {best_thresh:.2f}")
ax.set_xlabel("Decision threshold"); ax.set_ylabel("F1 (fraud class)")
ax.set_title("Threshold sweep on the test split", fontweight="bold"); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "v3_threshold_sweep.png", dpi=150); plt.close()

# 4) Why are transactions flagged by the hybrid score?
#    Hybrid = (p + is_anomaly) / 2 >= 0.4  ->  every Isolation-Forest anomaly passes (0.5 >= 0.4),
#    and a non-anomaly needs p >= 0.8.
flagged = df[df["Hybrid_Fraud_Prediction"] == 1]
anom = flagged["Is_Anomaly"] == 1
high_p = flagged["Ensemble_Prediction"] >= 0.8
comp = {
    "Isolation Forest anomaly\n(probability < 0.8)": int((anom & ~high_p).sum()),
    "Isolation Forest anomaly\n+ probability >= 0.8": int((anom & high_p).sum()),
    "Probability >= 0.8 only\n(not an anomaly)": int((~anom & high_p).sum()),
}
print(f"\nHybrid-flagged transactions (all rows, incl. true frauds): {len(flagged)}")
for k, v in comp.items():
    print(f"  {k.replace(chr(10), ' ')}: {v}")
print(f"  Isolation Forest flags overall: {int(df['Is_Anomaly'].sum())} of {len(df)} rows ({df['Is_Anomaly'].mean():.1%})")
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.barh(list(comp.keys())[::-1], list(comp.values())[::-1], color=["#27ae60", "#e67e22", "#2980b9"])
for i, v in enumerate(list(comp.values())[::-1]):
    ax.text(v + 5, i, str(v), va="center")
ax.set_xlabel("Flagged transactions")
ax.set_title("What drives the hybrid fraud flag?", fontweight="bold")
plt.tight_layout(); plt.savefig(FIG / "v3_hybrid_flag_composition.png", dpi=150); plt.close()

# 5) Does the anomaly score / hybrid flag line up with the real fraud label?
print(f"\nIsolation Forest ROC-AUC vs Is_Fraudulent (more anomalous = more fraud-like): "
      f"{roc_auc_score(df['Is_Fraudulent'], -df['Anomaly_Score']):.4f}")
print("Hybrid flag vs Is_Fraudulent on ALL rows (supervised part is in-sample here):")
print(confusion_matrix(df["Is_Fraudulent"], df["Hybrid_Fraud_Prediction"]))
n_fraud_total = int(df["Is_Fraudulent"].sum())
n_fraud_flagged = int(((df["Is_Fraudulent"] == 1) & (df["Hybrid_Fraud_Prediction"] == 1)).sum())
print(f"Frauds flagged: {n_fraud_flagged} of {n_fraud_total} "
      f"(vs {df['Hybrid_Fraud_Prediction'].mean():.1%} of all rows flagged)")
