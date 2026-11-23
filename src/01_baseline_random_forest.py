"""
v1 - Baseline: Random Forest on the cleaned transactions.

NOTE: this baseline uses Risk_Score as a feature. Risk_Score perfectly separates the
fraud label in this dataset (see 04_data_profile_and_leakage_audit.py), so its scores
are inflated by label leakage. Kept as the starting point of the project history.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "complete_dirty_bank_transactions.csv"
DATA_CLEAN = ROOT / "data" / "processed" / "cleaned_transactions.csv"
OUT = ROOT / "outputs"
FIG = ROOT / "assets" / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# Load cleaned data
df = pd.read_csv(DATA_CLEAN)

# Clean and convert monetary columns
money_cols = ['Loan_Amount', 'Account_Balance', 'Recent_Transaction_Amount', 'Avg_Monthly_Transaction']

for col in money_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(r'[^\d\.]', '', regex=True)  # Remove symbols/words
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values in monetary columns
df.dropna(subset=money_cols, inplace=True)
print("Remaining rows after cleaning:", len(df))

# Ensure Transaction_Time is in datetime format
df['Transaction_Time'] = pd.to_datetime(df['Transaction_Time'])

# Extract Hour if not already there
if 'Hour' not in df.columns:
    df['Hour'] = df['Transaction_Time'].dt.hour

# Create Time_Of_Day
def time_of_day(hour):
    if 0 <= hour < 6:
        return 'Night'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

df['Time_Of_Day'] = df['Hour'].apply(time_of_day)

# Create Weekday from Transaction_Time
df['Weekday'] = df['Transaction_Time'].dt.day_name()

# Create High_Value_Flag
df['High_Value_Flag'] = (
    df['Recent_Transaction_Amount'] > 1.5 * df['Avg_Monthly_Transaction']
).astype(int)

# Encode categorical features
categorical_cols = ['Gender', 'Account_Type', 'Transaction_Type', 'Pay_On_Time',
                    'Transaction_Location', 'Device_Type', 'Business_Type', 'Time_Of_Day', 'Weekday']

for col in categorical_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])

# Define features and target
features = [
    'Age', 'Gender', 'Account_Balance', 'Credit_Score', 'Loan_Amount',
    'Transaction_Type', 'Pay_On_Time', 'Recent_Transaction_Amount',
    'Avg_Monthly_Transaction', 'Device_Type', 'Account_Age_Years',
    'Risk_Score', 'High_Value_Flag', 'Hour', 'Weekday'
]
X = df[features]
y = df['Is_Fraudulent']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))

# Predict on full dataset
df['Model_Prediction'] = model.predict(X)

# Filter: model says fraud but bank didn’t
missed_frauds = df[(df['Is_Fraudulent'] == 0) & (df['Model_Prediction'] == 1)]

# Export for Power BI
missed_frauds.to_csv(OUT / 'missed_frauds_v1_random_forest.csv', index=False)
