import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt

# Load data and model
X_test = np.load('C:/RPM/data/X_test_features.npy')
y_test = np.load('C:/RPM/data/y_test.npy')
rf = joblib.load('C:/RPM/models/rf_classifier_tuned.pkl')

# Evaluate
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)
auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
fpr = cm.sum(axis=0) - np.diag(cm)
total = cm.sum(axis=0)
fpr_rate = np.mean(fpr / total) * 100

print("Final Test AUROC:", auc)
print("Final Test F1-score:", f1)
print("Final Test FPR (%):", fpr_rate)
print("Confusion Matrix:\n", cm)

# Plot ROC curves
fig, ax = plt.subplots()
for i in range(3):  # 3 classes
    RocCurveDisplay.from_predictions(
        (y_test == i).astype(int),
        y_pred_proba[:, i],
        name=f'Class {i}',
        ax=ax
    )
plt.savefig('C:/RPM/data/roc_curve.png')
plt.close()

# Save metrics
with open('C:/RPM/data/metrics.txt', 'w') as f:
    f.write(f"AUROC: {auc}\nF1-score: {f1}\nFPR: {fpr_rate}%")