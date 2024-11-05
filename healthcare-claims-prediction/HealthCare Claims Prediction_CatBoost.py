""" 0 . Import"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

#_______________________________________________________________________________________________________#

""" 1. Read all data sets """

# 1(a). Read: train.csv -> train_df
train_df = pd.read_csv(r"C:\Users\info\Desktop\BAN 693\healthcare-claims-prediction\train.csv")
print("Train CSV Data:")
train_df.head()

# 1(b). Read: patient_data_train.json -> patient_data_train_df
with open(r"C:\Users\info\Desktop\BAN 693\healthcare-claims-prediction\patient_data_train.json", 'r') as f:
    patient_data_train = json.load(f)

patient_data_train_df = pd.DataFrame(patient_data_train)
print("\nPatient Data (Train) JSON Data:")
print(patient_data_train_df.head())

# 1(c). Read: patient_data_test.json -> patient_data_test_df
with open(r"C:\Users\info\Desktop\BAN 693\healthcare-claims-prediction\patient_data_test.json", 'r') as f:
    patient_data_test = json.load(f)
patient_data_test_df = pd.DataFrame(patient_data_test)
print("\nPatient Data (Test) JSON Data:")
patient_data_test_df.head()
#patient_data_test_df.columns

# 1(d). Read: sample_submission.csv -> sample_submission_df
sample_submission_df = pd.read_csv(r"C:\Users\info\Desktop\BAN 693\healthcare-claims-prediction\sample_submission.csv")
sample_submission_df.head()

#_______________________________________________________________________________________________________#

"""2. Combine training data, and take a quick look """

# Combine: train.csv + patient_data_train.json - by "PatientID" -> train_data
train_data = pd.merge(train_df, patient_data_train_df, on="PatientID", how="inner")
train_data.head()

# Explore dataset info
print(train_data.info())
print(train_data.isnull().sum())
print(train_data.columns)
print(train_data.describe())

"""
Conclusion:
- No null.
- Total 6 column * 14629 rows (patients).
- Col name: 'PatientID', 'TotalClaims', 'Sex', 'Age', 'Conditions','Out patient costs'.
- Patient Age from 66 to 84, with a mean about 75. 
"""

#_______________________________________________________________________________________________________#

"""3(a). Deal with: Chronic "Conditions"""

# Define chronic_conditions to see all conditions existing in this dataset
chronic_conditions = set()
for condition in train_data['Conditions']:
    if isinstance(condition, dict):
        chronic_conditions.update(condition.keys())
print(chronic_conditions)

""" 
Conclusion:
There are 4 Conditions:
- AT: Arthritis/Atherosclerosis
- HT: Hypertension
- HD: Heart Disease
- DB: Diabetes
"""

# Visual: How severe for all the conditions?
condition_severity_distribution = {'AT': [], 'HT': [], 'HD': [], 'DB': []}
for condition in train_data['Conditions']:
    if isinstance(condition, dict):
        for cond, severity in condition.items():
            if cond in condition_severity_distribution:
                condition_severity_distribution[cond].append(severity)
                
for cond, severity_list in condition_severity_distribution.items():
    print(f"{cond}:", pd.Series(severity_list).value_counts().sort_index())

fig, ax = plt.subplots(2, 2, figsize=(12, 10))
condition_keys = list(condition_severity_distribution.keys())
colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99']

for i, cond in enumerate(condition_keys):
    row, col = i // 2, i % 2
    severity_counts = pd.Series(condition_severity_distribution[cond]).value_counts().sort_index()
    ax[row, col].bar(
        x=severity_counts.index,
        height=severity_counts.values,
        color=colors[i],
        alpha=0.7)
    ax[row, col].set_title(f'Severity Distribution for {cond}')
    ax[row, col].set_xlabel('Severity Level')
    ax[row, col].set_ylabel('Count')
    ax[row, col].set_xticks(range(1, 6))
plt.show()

# Split Chronic Conditions into 4 cols: AT, HT, HD, DB
def split_conditions(condition_dict):
    severity_columns = {'AT': 0, 'HT': 0, 'HD': 0, 'DB': 0}
    if isinstance(condition_dict, dict):
        for condition, severity in condition_dict.items():
            if condition in severity_columns:
                severity_columns[condition] = severity
    return pd.Series(severity_columns)

# Wrap up and Check
severity_df = train_data['Conditions'].apply(split_conditions)
train_data = pd.concat([train_data, severity_df], axis=1)
train_data.drop('Conditions', axis=1, inplace=True)
train_data.head()
train_data.columns


"""3(b). Deal with: Out patient cost"""

# How many years we got?
year = set()
for years in train_data['Out patient costs']:
    if isinstance(years, dict):
        year.update(years.keys())
print(year)

'''Conculsion: patient cost data from 2019 to 2023'''

''' NOTE: SUM? Sep cols? Growth Rate? standize?'''

# Separate yearly cost columns
def split_outpatient_costs(cost_dict):
    costs_columns = {'2019Cost': 0, '2020Cost': 0, '2021Cost': 0, '2022Cost': 0, '2023Cost': 0}
    if isinstance(cost_dict, dict):
        for year, cost in cost_dict.items():
            if f"{year}Cost" in costs_columns:
                costs_columns[f"{year}Cost"] = cost
    return pd.Series(costs_columns)

outpatient_costs_df = train_data['Out patient costs'].apply(split_outpatient_costs)
train_data = pd.concat([train_data, outpatient_costs_df], axis=1)
train_data.drop('Out patient costs', axis=1, inplace=True)

# Calculate TotalCost column
train_data['TotalCost'] = train_data[['2019Cost', '2020Cost', 
                                      '2021Cost', '2022Cost', 
                                      '2023Cost']].sum(axis=1)

train_data.head()
train_data.columns
train_data.dtypes

#_______________________________________________________________________________________________________#

"""4. Modelling"""
X = train_data.drop(['PatientID', 'TotalClaims', 'TotalCost'], axis=1)
y = train_data['TotalClaims']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check
X.columns
X.head()

#_______________________________________________________________________________________________________#
# 2.1 CatBoost, all factors
cat_model = CatBoostRegressor(random_state=42, verbose=0)
cat_model.fit(X_train, y_train, cat_features=['Sex'])

# Train Error
y_train_pred_cat = cat_model.predict(X_train)
train_mse_cat = mean_squared_error(y_train, y_train_pred_cat)
train_mae_cat = mean_absolute_error(y_train, y_train_pred_cat)
print(f"Train MSE: {train_mse_cat:.2f}, MAE: {train_mae_cat:.2f}")

# Validation Error
y_val_pred_cat = cat_model.predict(X_test)
val_mse_cat = mean_squared_error(y_test, y_val_pred_cat)
val_mae_cat = mean_absolute_error(y_test, y_val_pred_cat)
print(f"Validation MSE: {val_mse_cat:.2f}, MAE: {val_mae_cat:.2f}")

# MAPE
val_mape_cat = (abs(y_test - y_val_pred_cat) / y_test).mean() * 100
print(f"Validation MAPE: {val_mape_cat:.2f}%")

#_______________________________________________________________________________________________________#

"""5. Same with Test_data"""

# Condition
def split_conditions(condition_dict):
    severity_columns = {'AT': 0, 'HT': 0, 'HD': 0, 'DB': 0}
    if isinstance(condition_dict, dict):
        for condition, severity in condition_dict.items():
            if condition in severity_columns:
                severity_columns[condition] = severity
    return pd.Series(severity_columns)

severity_df_test = patient_data_test_df['Conditions'].apply(split_conditions)
patient_data_test_df = pd.concat([patient_data_test_df, severity_df_test], axis=1)
patient_data_test_df.drop('Conditions', axis=1, inplace=True)

# Out patient costs
def split_outpatient_costs(cost_dict):
    costs_columns = {'2019Cost': 0, '2020Cost': 0, '2021Cost': 0, '2022Cost': 0, '2023Cost': 0}
    if isinstance(cost_dict, dict):
        for year, cost in cost_dict.items():
            if f"{year}Cost" in costs_columns:
                costs_columns[f"{year}Cost"] = cost
    return pd.Series(costs_columns)

outpatient_costs_df_test = patient_data_test_df['Out patient costs'].apply(split_outpatient_costs)
patient_data_test_df = pd.concat([patient_data_test_df, outpatient_costs_df_test], axis=1)
patient_data_test_df.drop('Out patient costs', axis=1, inplace=True)

# Total Cost
patient_data_test_df['TotalCost'] = patient_data_test_df[['2019Cost', '2020Cost', '2021Cost', '2022Cost', '2023Cost']].sum(axis=1)

# Sex
# patient_data_test_df = pd.get_dummies(patient_data_test_df, columns=['Sex'], drop_first=True)

# Save and check test_data
test_data = patient_data_test_df.copy()
test_data.head()
test_data.columns

# Drop PatientID
X_test_submission = test_data.drop(['PatientID'], axis=1)

#_______________________________________________________________________________________________________#
"""6. Submission - round 2.0"""
# CatBoost - 1.4 submission
y_pred_cat = cat_model.predict(X_test_submission)
sample_submission_df['TotalClaims'] = y_pred_cat
sample_submission_df.to_csv(r"C:\Users\info\Desktop\BAN 693\healthcare-claims-prediction\submission_catboost_2.2.csv", index=False)
