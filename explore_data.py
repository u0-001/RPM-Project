import pandas as pd

# Load vital signs data
data = pd.read_csv('C:/RPM/data/eicu-crd-demo/vitalPeriodic.csv')

# Display first 5 rows
print(data.head())

# Display column names
print(data.columns)

# Check for missing values
print(data.isnull().sum())

# Basic statistics
print(data.describe())