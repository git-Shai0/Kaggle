import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('C:\\Users\\USER\\Desktop\\projects\\Capstone project\\ccdata\\CC GENERAL.csv')

# Drop ID for preprocessing (keep aside)
cust_id = df['CUST_ID']
df = df.drop(columns=['CUST_ID'])

# Inspect
print(df.info())
print("Missing values:\n", df.isnull().sum())

# Impute numeric missing values with median
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Identify monetary variables for scaling
money_cols = ['BALANCE','PURCHASES','CASH_ADVANCE','CREDIT_LIMIT',
              'PAYMENTS','MINIMUM_PAYMENTS']

scaler = StandardScaler()
df_imputed[money_cols] = scaler.fit_transform(df_imputed[money_cols])

print("✅ Missing handled and monetary features normalized.")

# Avoid divide-by-zero
df_imputed['avg_monthly_spend'] = np.where(df_imputed['TENURE']>0,
                                           df_imputed['PURCHASES']/df_imputed['TENURE'], 0)

df_imputed['installment_share'] = np.where(df_imputed['PURCHASES']>0,
                                           df_imputed['INSTALLMENTS_PURCHASES']/df_imputed['PURCHASES'], 0)

df_imputed['oneoff_share'] = np.where(df_imputed['PURCHASES']>0,
                                      df_imputed['ONEOFF_PURCHASES']/df_imputed['PURCHASES'], 0)

df_imputed['utilization_rate'] = np.where(df_imputed['CREDIT_LIMIT']>0,
                                          df_imputed['BALANCE']/df_imputed['CREDIT_LIMIT'], 0)

df_imputed['payment_to_balance_ratio'] = np.where(df_imputed['BALANCE']>0,
                                                  df_imputed['PAYMENTS']/df_imputed['BALANCE'], 0)

df_imputed['purchase_to_cash_ratio'] = df_imputed['PURCHASES'] / (df_imputed['CASH_ADVANCE'] + 1)

print(df_imputed[['avg_monthly_spend','installment_share','oneoff_share',
                  'utilization_rate','payment_to_balance_ratio','purchase_to_cash_ratio']].head())


# Compute percentiles for dynamic cutoffs
q3_spend = df_imputed['avg_monthly_spend'].quantile(0.75)
q3_cash = df_imputed['CASH_ADVANCE'].quantile(0.75)

def label_segment(row):
    if row['TENURE'] <= 12:
        return 'New Customer'
    elif row['avg_monthly_spend'] > q3_spend:
        return 'High Spender'
    elif row['installment_share'] > 0.6:
        return 'Installment Heavy'
    elif (row['utilization_rate'] > 0.8) and (row['payment_to_balance_ratio'] < 0.3):
        return 'Revolver'
    elif row['CASH_ADVANCE'] > q3_cash:
        return 'Cash Advance User'
    else:
        return 'Standard'

df_imputed['Segment'] = df_imputed.apply(label_segment, axis=1)
df_imputed['CUST_ID'] = cust_id

# Summary count
print(df_imputed['Segment'].value_counts())
