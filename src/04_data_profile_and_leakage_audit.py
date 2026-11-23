"""
Data profile + label-leakage audit.

Questions answered
------------------
1. How imbalanced and how "dirty" is the raw table?
2. Does any single feature separate fraud from non-fraud on its own?
3. How much of the baseline model's performance comes from Risk_Score alone?
4. Do the other features (hour, transaction type, device, ...) carry any fraud signal?

Outputs: figures in assets/figures/, numbers printed to stdout (saved in outputs/logs/).
Run from anywhere:  python src/04_data_profile_and_leakage_audit.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "complete_dirty_bank_transactions.csv"
DATA_CLEAN = ROOT / "data" / "processed" / "cleaned_transactions.csv"
FIG = ROOT / "assets" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_style("whitegrid")
SEED = 42

raw = pd.read_csv(DATA_RAW)
raw["Transaction_Time"] = pd.to_datetime(raw["Transaction_Time"])
raw["Hour"] = raw["Transaction_Time"].dt.hour
y = raw["Is_Fraudulent"]

# ------------------------------------------------------------------ 1. basic profile
print("=== BASIC PROFILE ===")
print(f"Rows: {len(raw):,} | customers: {raw['Customer_ID'].nunique():,} | columns: {raw.shape[1] - 1}")
print(f"Fraud: {int(y.sum())} ({y.mean():.2%}) | legit: {int((y == 0).sum())}")
print(f"Distinct transaction dates: {raw['Transaction_Time'].dt.date.nunique()} "
      f"({raw['Transaction_Time'].dt.date.iloc[0]})")
txn_per_cust = raw.groupby("Customer_ID").size()
print(f"Transactions per customer: min={txn_per_cust.min()}, median={int(txn_per_cust.median())}, max={txn_per_cust.max()}")
nulls = raw.isna().sum()
nulls = nulls[nulls > 0]
text_loans = int(pd.to_numeric(raw["Loan_Amount"], errors="coerce").isna().sum() - raw["Loan_Amount"].isna().sum())
print("Missing values per column:", nulls.to_dict())
print(f"Loan_Amount cells holding text instead of a number: {text_loans}")
print(f"Rows with at least one missing value: {int(raw.isna().any(axis=1).sum())}")
loc_state = raw["Transaction_Location"].str.split(",").str[1].str.strip()
print(f"Transaction_Location state != customer State: {(loc_state != raw['State']).mean():.1%} of rows")

# Figure: class balance + missingness
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
counts = y.value_counts().sort_index()
bars = ax[0].bar(["Legitimate", "Fraudulent"], counts.values, color=["#2980b9", "#c0392b"])
for b, v in zip(bars, counts.values):
    ax[0].text(b.get_x() + b.get_width() / 2, v + 60, f"{v:,}\n({v / len(raw):.1%})", ha="center")
ax[0].set_ylim(0, counts.max() * 1.18)
ax[0].set_title("Severe class imbalance", fontweight="bold")
ax[0].set_ylabel("Transactions")
miss = nulls.copy()
miss["Loan_Amount (text, e.g. 'Ten Thousand')"] = text_loans
miss = miss.sort_values()
ax[1].barh(miss.index, miss.values, color="#e67e22")
for i, v in enumerate(miss.values):
    ax[1].text(v + 1, i, str(int(v)), va="center")
ax[1].set_title("Dirty values in the raw file", fontweight="bold")
ax[1].set_xlabel("Affected cells")
plt.tight_layout()
plt.savefig(FIG / "data_quality_overview.png", dpi=150)
plt.close()

# ------------------------------------------------------------------ 2. single-feature separability
print("\n=== SINGLE-FEATURE ROC-AUC (direction-agnostic, 0.5 = no signal) ===")
num = raw.copy()
num["Loan_Amount"] = pd.to_numeric(num["Loan_Amount"], errors="coerce")
candidates = ["Risk_Score", "Account_Age_Years", "Avg_Monthly_Transaction", "Recent_Transaction_Amount",
              "Age", "Credit_Score", "Account_Balance", "Loan_Amount", "Hour"]
aucs = {}
for c in candidates:
    m = num[c].notna()
    a = roc_auc_score(y[m], num.loc[m, c])
    aucs[c] = max(a, 1 - a)
    print(f"  {c:28s} {aucs[c]:.3f}")
s = pd.Series(aucs).sort_values()
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.barh(s.index, s.values, color=["#c0392b" if k == "Risk_Score" else "#7f8c8d" for k in s.index])
ax.axvline(0.5, ls="--", color="black", lw=1)
ax.text(0.505, -0.75, "no signal (0.5)", fontsize=9)
for i, v in enumerate(s.values):
    ax.text(v + 0.008, i, f"{v:.3f}", va="center")
ax.set_xlim(0.4, 1.08)
ax.set_xlabel("Single-feature ROC-AUC (direction-agnostic)")
ax.set_title("Only Risk_Score separates fraud; every other feature is at chance", fontweight="bold")
plt.tight_layout()
plt.savefig(FIG / "single_feature_auc.png", dpi=150)
plt.close()

# ------------------------------------------------------------------ 3. Risk_Score vs the label
print("\n=== RISK_SCORE BY CLASS ===")
rs = raw.groupby("Is_Fraudulent")["Risk_Score"].agg(["min", "median", "max"])
print(rs.round(3).to_string())
max_legit = raw.loc[y == 0, "Risk_Score"].max()
min_fraud = raw.loc[y == 1, "Risk_Score"].min()
pure = ((raw["Risk_Score"] > 0.75).astype(int) == y).mean()
print(f"max Risk_Score among legit = {max_legit:.3f}; min among fraud = {min_fraud:.3f}")
print(f"Rule 'Risk_Score > 0.75' reproduces Is_Fraudulent for {pure:.2%} of all rows")

fig, ax = plt.subplots(figsize=(9, 4.6))
bins = np.linspace(0, 1, 51)
ax.hist(raw.loc[y == 0, "Risk_Score"], bins=bins, color="#2980b9", alpha=0.75, label="Legitimate (n=4,946)")
ax.hist(raw.loc[y == 1, "Risk_Score"], bins=bins, color="#c0392b", alpha=0.9, label="Fraudulent (n=55)")
ax.axvline(0.75, color="black", ls="--")
ax.text(0.755, ax.get_ylim()[1] * 0.85, "0.75 cut-off:\nno legit row above,\nno fraud row below", fontsize=9)
ax.set_xlabel("Risk_Score")
ax.set_ylabel("Transactions")
ax.set_title("Risk_Score perfectly encodes the fraud label", fontweight="bold")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(FIG / "risk_score_leakage.png", dpi=150)
plt.close()

# ------------------------------------------------------------------ 4. leakage ablation (same RF as the baseline)
print("\n=== LEAKAGE ABLATION: Random Forest with vs without Risk_Score ===")
c = pd.read_csv(DATA_CLEAN)
for col in ["Loan_Amount", "Account_Balance", "Recent_Transaction_Amount", "Avg_Monthly_Transaction"]:
    c[col] = pd.to_numeric(c[col].astype(str).str.replace(r"[^\d\.]", "", regex=True), errors="coerce")
c = c.dropna(subset=["Loan_Amount", "Account_Balance", "Recent_Transaction_Amount", "Avg_Monthly_Transaction"])
c["Transaction_Time"] = pd.to_datetime(c["Transaction_Time"])
c["Hour"] = c["Transaction_Time"].dt.hour
c["High_Value_Flag"] = (c["Recent_Transaction_Amount"] > 1.5 * c["Avg_Monthly_Transaction"]).astype(int)
for col in ["Gender", "Transaction_Type", "Pay_On_Time", "Device_Type"]:
    c[col] = c[col].astype("category").cat.codes
base_features = ["Age", "Gender", "Account_Balance", "Credit_Score", "Loan_Amount", "Transaction_Type",
                 "Pay_On_Time", "Recent_Transaction_Amount", "Avg_Monthly_Transaction", "Device_Type",
                 "Account_Age_Years", "High_Value_Flag", "Hour"]
yc = c["Is_Fraudulent"]
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
results = {}
for name, feats in {"With Risk_Score": base_features + ["Risk_Score"], "Without Risk_Score": base_features}.items():
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=-1)
    cvres = cross_validate(rf, c[feats], yc, cv=cv, scoring=["roc_auc", "average_precision"])
    results[name] = {
        "roc_auc": (cvres["test_roc_auc"].mean(), cvres["test_roc_auc"].std()),
        "pr_auc": (cvres["test_average_precision"].mean(), cvres["test_average_precision"].std()),
    }
    print(f"  {name:20s} ROC-AUC {results[name]['roc_auc'][0]:.3f} +/- {results[name]['roc_auc'][1]:.3f} | "
          f"PR-AUC {results[name]['pr_auc'][0]:.3f} +/- {results[name]['pr_auc'][1]:.3f}")
print(f"  (cleaned data: {len(c)} rows, {int(yc.sum())} frauds, prevalence {yc.mean():.3%}; "
      f"5-fold x 3 repeats stratified CV)")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for a, key, title in zip(ax, ["roc_auc", "pr_auc"], ["ROC-AUC", "PR-AUC (average precision)"]):
    names = list(results)
    vals = [results[n][key][0] for n in names]
    errs = [results[n][key][1] for n in names]
    a.bar(names, vals, yerr=errs, color=["#c0392b", "#2980b9"], capsize=5)
    for i, v in enumerate(vals):
        a.text(i, v + 0.02, f"{v:.3f}", ha="center")
    a.set_ylim(0, 1.12)
    a.set_title(title, fontweight="bold")
ax[0].axhline(0.5, ls="--", color="grey")
ax[0].text(0.5, 0.52, "chance", ha="center", fontsize=9, color="grey")
ax[1].axhline(yc.mean(), ls="--", color="grey")
ax[1].text(0.5, yc.mean() + 0.02, f"no-skill = {yc.mean():.3f}", ha="center", fontsize=9, color="grey")
plt.suptitle("Random Forest, stratified 5-fold x 3 repeats: the whole result is Risk_Score", fontweight="bold")
plt.tight_layout()
plt.savefig(FIG / "leakage_ablation.png", dpi=150)
plt.close()

# ------------------------------------------------------------------ 5. any signal in the categorical features?
print("\n=== FRAUD RATE BY CATEGORY (Wilson 95% intervals) ===")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return centre - half, centre + half


overall = y.mean()
raw["Hour_Block"] = pd.cut(raw["Hour"], [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"])
fig, ax = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
for a, col in zip(ax, ["Transaction_Type", "Device_Type", "Account_Type", "Hour_Block"]):
    g = raw.groupby(col, observed=True)["Is_Fraudulent"].agg(["sum", "count"])
    rates = g["sum"] / g["count"]
    ci = np.array([wilson(k, n) for k, n in zip(g["sum"], g["count"])])
    a.bar(g.index.astype(str), rates * 100, color="#8e44ad", alpha=0.85)
    a.errorbar(range(len(g)), rates * 100, yerr=[(rates - ci[:, 0]) * 100, (ci[:, 1] - rates) * 100],
               fmt="none", ecolor="black", capsize=3)
    a.axhline(overall * 100, ls="--", color="grey", label=f"overall rate {overall:.2%}")
    a.legend(loc="upper right", fontsize=8)
    a.set_title(col.replace("_", " "), fontweight="bold")
    a.tick_params(axis="x", rotation=30)
    print(f"  {col}: " + ", ".join(f"{i}={r:.2%} (n={n})" for i, r, n in zip(g.index, rates, g["count"])))
ax[0].set_ylabel("Fraud rate (%)")
plt.suptitle("Fraud rate by category (error bars: Wilson 95% CI; dashed: overall rate)", fontweight="bold")
plt.tight_layout()
plt.savefig(FIG / "fraud_rate_by_category.png", dpi=150)
plt.close()
print("\nDone. Figures written to", FIG)
