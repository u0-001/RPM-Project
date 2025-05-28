import pandas as pd

# Load raw data
data = pd.read_csv('C:/RPM/data/eicu-crd-demo/vitalPeriodic.csv')

# Print column names
print("Columns:", data.columns.tolist())

# Check systemicsystolic and systemicdiastolic
print("\nSystemicsystolic summary:")
print(data['systemicsystolic'].describe())
print(data['systemicsystolic'].isnull().sum(), "missing values")

print("\nSystemicdiastolic summary:")
print(data['systemicdiastolic'].describe())
print(data['systemicdiastolic'].isnull().sum(), "missing values")