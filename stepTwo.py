import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import the data file here as well
df = pd.read_csv('C:\\Users\\USER\\Desktop\\projects\\Capstone project\\ccdata\\CC GENERAL.csv')

# Sort and view extremes
df_imputed = df.copy()
# --- Derived Features ---
# Average monthly spend = Total Purchases / Tenure
df_imputed['avg_monthly_spend'] = df_imputed['PURCHASES'] / df_imputed['TENURE']

# Percentage of installment vs one-off purchases
df_imputed['installment_pct'] = (df_imputed['INSTALLMENTS_PURCHASES'] / df_imputed['PURCHASES']).replace([float('inf'), float('nan')], 0)

# Utilization of credit limit
df_imputed['utilization_rate'] = (df_imputed['BALANCE'] / df_imputed['CREDIT_LIMIT']).replace([float('inf'), float('nan')], 0)

# Payment to balance ratio
df_imputed['payment_to_balance_ratio'] = (df_imputed['PAYMENTS'] / df_imputed['BALANCE']).replace([float('inf'), float('nan')], 0)

print(" Derived features created successfully!")

top_spenders = df_imputed.nlargest(10, 'avg_monthly_spend')[['CUST_ID','avg_monthly_spend','utilization_rate','payment_to_balance_ratio']]
low_spenders = df_imputed.nsmallest(10, 'avg_monthly_spend')[['CUST_ID','avg_monthly_spend','utilization_rate','payment_to_balance_ratio']]

print("Top 10 Spenders:\n", top_spenders)
print("\nLowest 10 Spenders:\n", low_spenders)

# Visualize distribution
plt.figure(figsize=(7,4))
sns.histplot(df_imputed['avg_monthly_spend'], bins=30, kde=True)
plt.title("Distribution of Average Monthly Spend")
plt.xlabel("Average Monthly Spend (normalized)")
plt.show()

# Compute dominance type
# Derived feature: purchase behavior shares
df_imputed['installment_share'] = df_imputed['INSTALLMENTS_PURCHASES'] / df_imputed['PURCHASES']
df_imputed['oneoff_share'] = df_imputed['ONEOFF_PURCHASES'] / df_imputed['PURCHASES']

# Replace infinities and NaNs (for customers with no purchases)
df_imputed['installment_share'] = df_imputed['installment_share'].replace([np.inf, -np.inf], 0).fillna(0)
df_imputed['oneoff_share'] = df_imputed['oneoff_share'].replace([np.inf, -np.inf], 0).fillna(0)

# Identify dominant purchase type
df_imputed['purchase_type'] = np.where(
    df_imputed['oneoff_share'] > df_imputed['installment_share'],
    'One-off',
    'Installment'
)


df_imputed['dominant_purchase_type'] = np.where(
    df_imputed['oneoff_share'] > df_imputed['installment_share'], 'One-off', 'Installment')

sns.countplot(x='dominant_purchase_type', data=df_imputed)
plt.title("Dominant Purchase Type per Customer")
plt.xlabel("Purchase Type")
plt.ylabel("Number of Customers")
plt.show()

# Average spending by type
avg_spend_by_type = df_imputed.groupby('dominant_purchase_type')['avg_monthly_spend'].mean()
print("Average monthly spend by dominant type:\n", avg_spend_by_type)

# Categorize tenure and credit limit into bins
df_imputed['tenure_group'] = pd.cut(df_imputed['TENURE'], bins=[0,12,24,36,48,60], labels=['<1yr','1-2yr','2-3yr','3-4yr','4-5yr'])
df_imputed['limit_group'] = pd.qcut(df_imputed['CREDIT_LIMIT'], q=4, labels=['Low','Medium','High','Very High'])

# Average spend and repayment per group
tenure_summary = df_imputed.groupby('tenure_group')[['avg_monthly_spend','payment_to_balance_ratio','utilization_rate']].mean().round(2)
limit_summary = df_imputed.groupby('limit_group')[['avg_monthly_spend','payment_to_balance_ratio','utilization_rate']].mean().round(2)

print("Spend & Repayment by Tenure:\n", tenure_summary)
print("\nSpend & Repayment by Credit Limit:\n", limit_summary)

# Visualization
fig, ax = plt.subplots(1,2, figsize=(12,4))
sns.barplot(x='tenure_group', y='avg_monthly_spend', data=df_imputed, ax=ax[0])
ax[0].set_title('Average Spend by Tenure')

sns.barplot(x='limit_group', y='avg_monthly_spend', data=df_imputed, ax=ax[1])
ax[1].set_title('Average Spend by Credit Limit')
plt.show()

# Compute heavy cash-advance flag
q75_cash = df_imputed['CASH_ADVANCE'].quantile(0.75)
df_imputed['heavy_cash_user'] = np.where(df_imputed['CASH_ADVANCE'] > q75_cash, 1, 0)

# Summary comparison
cash_summary = df_imputed.groupby('heavy_cash_user')[['CASH_ADVANCE','utilization_rate','payment_to_balance_ratio','PRC_FULL_PAYMENT']].mean().round(2)
print(cash_summary)

# Visualize
plt.figure(figsize=(6,4))
sns.boxplot(x='heavy_cash_user', y='CASH_ADVANCE', data=df_imputed)
plt.title("Cash Advance Distribution: Heavy vs Normal Users")
plt.xlabel("Heavy Cash Advance User (1=Yes)")
plt.ylabel("Cash Advance Amount")
plt.show()

df_imputed.to_csv('cleaned_cc_data.csv', index=False)
print(" Cleaned dataset saved successfully as 'cleaned_cc_data.csv'")

df_imputed['customer_segment'] = pd.qcut(
    df_imputed['avg_monthly_spend'], 
    q=3, 
    labels=['Low', 'Medium', 'High']
)

df_imputed.to_csv('cleaned_cc_data.csv', index=False)
print("Customer segments added successfully and file saved.")