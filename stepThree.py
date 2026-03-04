import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned/imputed dataset from Step 2
df = pd.read_csv('cleaned_cc_data.csv')  # or df_imputed if saved directly

# Create spend categories
df['spend_category'] = pd.qcut(df['avg_monthly_spend'], q=3, labels=['Low', 'Medium', 'High'])

# Plot segment distribution
plt.figure(figsize=(7,5))
sns.countplot(data=df, x='spend_category', palette='viridis')
plt.title('Customer Segments by Monthly Spend')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.show()

plt.figure(figsize=(8,5))
sns.lineplot(data=df, x='TENURE', y='avg_monthly_spend', marker='o')
plt.title('Average Monthly Spend vs Tenure')
plt.xlabel('Customer Tenure (Months)')
plt.ylabel('Average Monthly Spend')
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x='BALANCE',
    y='PAYMENTS',
    hue='spend_category',
    alpha=0.7,
    palette='coolwarm'
)
plt.title('Balance vs Payment Behavior')
plt.xlabel('Balance')
plt.ylabel('Payments')
plt.legend(title='Spending Category')
plt.show()

plt.figure(figsize=(7,5))
sns.histplot(df['utilization_rate'], bins=20, kde=True, color='orange')
plt.title('Distribution of Credit Utilization Rates')
plt.xlabel('Utilization Rate (%)')
plt.ylabel('Number of Customers')
plt.show()

# 20% of customers contributing to 60% of purchases
top_20_percent = df.nlargest(int(0.2 * len(df)), 'avg_monthly_spend')
percent_purchases = (top_20_percent['avg_monthly_spend'].sum() / df['avg_monthly_spend'].sum()) * 100
print(f" Top 20% customers contribute to {percent_purchases:.1f}% of total purchases.")

# Cash advance risk
cash_advance_customers = df[df['CASH_ADVANCE'] > 0]
avg_overdue = cash_advance_customers['PAYMENTS'].mean() / cash_advance_customers['BALANCE'].mean()
print(f" Cash advance users have an average payment-to-balance ratio of {avg_overdue:.2f}")

# New customer growth
new_customers = df[df['TENURE'] < 12]
avg_new_spend = new_customers['avg_monthly_spend'].mean()
print(f" New customers spend on average ${avg_new_spend:.2f} but show fast growth.")
