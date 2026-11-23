"""Export the raw table as a star schema (2 dimensions + 1 fact) for Power BI."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "complete_dirty_bank_transactions.csv"
STAR = ROOT / "data" / "star_schema"
STAR.mkdir(parents=True, exist_ok=True)

# Load the main dataset
df = pd.read_csv(DATA_RAW)

# 1. dim_customer
dim_customer = df[[
    'Customer_ID', 'First_Name', 'Last_Name', 'Gender', 'Age', 'State',
    'Join_Date', 'Has_Loan', 'Loan_Amount', 'Business_Type', 'Pay_On_Time'
]].drop_duplicates()

# 2. dim_account
dim_account = df[[
    'Customer_ID', 'Account_Type', 'Account_Balance', 'Credit_Score'
]].drop_duplicates()

# 3. fact_transaction
fact_transaction = df[[
    'Customer_ID', 'Transaction_Time', 'Transaction_Type', 'Recent_Transaction_Amount',
    'Transaction_Location', 'Avg_Monthly_Transaction', 'Device_Type',
    'Account_Age_Years', 'Risk_Score', 'Is_Fraudulent'
]]

# Save all 3 as CSVs
dim_customer.to_csv(STAR / "dim_customer.csv", index=False)
dim_account.to_csv(STAR / "dim_account.csv", index=False)
fact_transaction.to_csv(STAR / "fact_transaction.csv", index=False)
