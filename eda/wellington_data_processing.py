"""
IEOR4742 - Wellington Management Anomaly Detection, Fall 2022

This is the data processing pipeline for the credit card data set.
Used for reading, feature engineering, and saving.
"""

import numpy as np
import pandas as pd
import os

# global variables
data_dir = "./data"
credit_subdir = os.path.join(data_dir, "credit")
sales_subdir = os.path.join(data_dir, "sales")
max_lag = 5

credit = pd.read_csv(os.path.join(data_dir, f"data4columbia_credit.csv"))
sales = pd.read_csv(os.path.join(data_dir, f"data4columbia_sales.csv"))

### Hardcoding trimming ###
## CREDIT ##
# add name to first column, 'symbol' to be consistent
credit.rename(columns={"Unnamed: 0": "symbol"}, inplace=True)

# Restructure data via pivot and fillna with 0
credit = credit.pivot_table(
    values="data", index="trans_date", columns="symbol")

# Drop any company that has more than 20% missing nans
b = credit.apply(lambda x: x.isna().sum() <= len(credit)*0.2)
credit = credit.loc[:, [col for col in b.loc[b].index]]

# Fill nan with 0
credit.fillna(0, inplace=True)

# Feature Engineering
# finite difference, first order
d1 = credit.diff()
d1.columns = [f"{col}_d1" for col in credit.columns]
# finite difference, second order
d2 = d1.diff()
d2.columns = [f"{col}_d2" for col in credit.columns]
# percentage change
#ret = pd.DataFrame(d1.values / credit.values) #! FAILS

ret = pd.DataFrame(d1.iloc[1:, :].values / credit.iloc[1:, :].values)

ret.columns = [f"{col}_ret" for col in credit.columns]
ret.index = credit.index

# cleaning
featured_credit = pd.concat([credit, d1, d2, ret], axis=1)
featured_credit.replace([np.inf, -np.inf], 0, inplace=True)
featured_credit.drop(index=[featured_credit.index[i]
                     for i in [0, 1, -1]], inplace=True)
featured_credit_filename = "featured_credit.csv"
featured_credit.to_csv(os.path.join(data_dir, featured_credit_filename))
