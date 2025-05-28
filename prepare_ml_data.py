import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load data
data = pd.read_csv('C:/RPM/data/combined_vitals.csv')
vital_columns = ['heartrate', 'sao2', 'respiration', 'systemicsystolic', 'systemicdiastolic']

# Balanced health status labels
def assign_label(row):
    if (row['heartrate'] < 0.3 or row['heartrate'] > 0.7 or
        row['sao2'] < 0.88 or
        row['respiration'] < 0.2 or row['respiration'] > 0.6 or
        row['systemicsystolic'] < 0.4 or row['systemicsystolic'] > 0.6):
        return 2  # Critical
    elif (row['heartrate'] > 0.5 or row['sao2'] < 0.94 or
          row['respiration'] > 0.4 or row['systemicsystolic'] > 0.5):
        return 1  # At-risk
    return 0  # Normal

data['health_status'] = data.apply(assign_label, axis=1)
print("Initial label distribution:\n", data['health_status'].value_counts())

# Create sequences
sequences = []
labels = []
for i in range(len(data) - 9):
    seq = data[vital_columns].iloc[i:i+10].values
    if len(seq) == 10:
        sequences.append(seq)
        labels.append(data['health_status'].iloc[i+9])
sequences = np.array(sequences)
labels = np.array(labels)

# Oversample minority class (normal)
smote = SMOTE(random_state=42)
sequences_flat = sequences.reshape(len(sequences), -1)  # Flatten for SMOTE
sequences_resampled, labels_resampled = smote.fit_resample(sequences_flat, labels)
sequences_resampled = sequences_resampled.reshape(-1, 10, 5)  # Reshape back

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    sequences_resampled, labels_resampled, test_size=0.3, random_state=42, stratify=labels_resampled
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Save splits
np.save('C:/RPM/data/X_train.npy', X_train)
np.save('C:/RPM/data/X_val.npy', X_val)
np.save('C:/RPM/data/X_test.npy', X_test)
np.save('C:/RPM/data/y_train.npy', y_train)
np.save('C:/RPM/data/y_val.npy', y_val)
np.save('C:/RPM/data/y_test.npy', y_test)

print("Shapes:", X_train.shape, X_val.shape, X_test.shape)
print("Label counts (train):", np.bincount(y_train))