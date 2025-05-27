import pandas as pd

# Load files
preprocessed = pd.read_csv('C:/RPM/data/preprocessed_vitals.csv')
synthetic = pd.read_csv('C:/RPM/data/synthetic_vitals.csv')
combined = pd.read_csv('C:/RPM/data/combined_vitals.csv')

# Print summaries
print("\nPreprocessed data:")
print("Rows:", len(preprocessed))
print("Columns:", preprocessed.columns.tolist())
print("Missing values:\n", preprocessed[['systemicsystolic', 'systemicdiastolic']].isnull().sum())
print("Sample BP:\n", preprocessed[['systemicsystolic', 'systemicdiastolic']].head())

print("\nSynthetic data:")
print("Rows:", len(synthetic))
print("Columns:", synthetic.columns.tolist())
print("Missing values:\n", synthetic[['systemicsystolic', 'systemicdiastolic']].isnull().sum())
print("Sample BP:\n", synthetic[['systemicsystolic', 'systemicdiastolic']].head())

print("\nCombined data:")
print("Rows:", len(combined))
print("Columns:", combined.columns.tolist())
print("Missing values:\n", combined[['systemicsystolic', 'systemicdiastolic']].isnull().sum())