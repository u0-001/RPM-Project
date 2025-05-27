import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('C:/RPM/data/combined_vitals.csv')
vital_columns = ['heartrate', 'sao2', 'respiration', 'systemicsystolic', 'systemicdiastolic']

# Simulate health status labels (based on thesis thresholds)
def assign_label(row):
    if (row['heartrate'] < 0.2 or row['heartrate'] > 0.8 or
        row['sao2'] < 0.9 or
        row['respiration'] < 0.1 or row['respiration'] > 0.7 or
        row['systemicsystolic'] < 0.3 or row['systemicsystolic'] > 0.7):
        return 2  # Critical
    elif (row['heartrate'] > 0.6 or row['sao2'] < 0.95 or
          row['respiration'] > 0.5 or row['systemicsystolic'] > 0.6):
        return 1  # At-risk
    return 0  # Normal

data['health_status'] = data.apply(assign_label, axis=1)
print("Label distribution:\n", data['health_status'].value_counts())

# Create sequences for LSTM (10 timesteps, 5 features)
sequences = []
labels = []
for i in range(0, len(data) - 9, 10):  # Step by 10 to avoid overlap
    seq = data[vital_columns].iloc[i:i+10].values
    if len(seq) == 10:  # Ensure full sequence
        sequences.append(seq)
        labels.append(data['health_status'].iloc[i+9])  # Label at last timestep
sequences = np.array(sequences)
labels = np.array(labels)

# Split data (70% train, 15% validation, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(sequences, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save splits
np.save('C:/RPM/data/X_train.npy', X_train)
np.save('C:/RPM/data/X_val.npy', X_val)
np.save('C:/RPM/data/X_test.npy', X_test)
np.save('C:/RPM/data/y_train.npy', y_train)
np.save('C:/RPM/data/y_val.npy', y_val)
np.save('C:/RPM/data/y_test.npy', y_test)

print("Shapes:", X_train.shape, X_val.shape, X_test.shape)
print("Label counts (train):", np.bincount(y_train))