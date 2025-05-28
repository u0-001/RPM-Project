import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import numpy as np
import joblib

# Load data
X_train = np.load('C:/RPM/data/X_train_features.npy')
y_train = np.load('C:/RPM/data/y_train.npy')
X_val = np.load('C:/RPM/data/X_val_features.npy')
y_val = np.load('C:/RPM/data/y_val.npy')
X_test = np.load('C:/RPM/data/X_test_features.npy')
y_test = np.load('C:/RPM/data/y_test.npy')

# Compute class weights
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

# Train Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2, class_weight=class_weights, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred_val = rf.predict(X_val)
y_pred_proba_val = rf.predict_proba(X_val)
auc = roc_auc_score(y_val, y_pred_proba_val, multi_class='ovr')
f1 = f1_score(y_val, y_pred_val, average='weighted')
cm = confusion_matrix(y_val, y_pred_val)
fpr = cm.sum(axis=0) - np.diag(cm)
total = cm.sum(axis=0)
fpr_rate = np.mean(fpr / total) * 100

print("Validation AUROC:", auc)
print("Validation F1-score:", f1)
print("Validation FPR (%):", fpr_rate)

# Test evaluation
y_pred_test = rf.predict(X_test)
y_pred_proba_test = rf.predict_proba(X_test)
auc_test = roc_auc_score(y_test, y_pred_proba_test, multi_class='ovr')
f1_test = f1_score(y_test, y_pred_test, average='weighted')
cm_test = confusion_matrix(y_test, y_pred_test)
fpr_test = cm_test.sum(axis=0) - np.diag(cm_test)
total_test = cm_test.sum(axis=0)
fpr_rate_test = np.mean(fpr_test / total_test) * 100

print("\nTest AUROC:", auc_test)
print("Test F1-score:", f1_test)
print("Test FPR (%):", fpr_rate_test)

# Save
joblib.dump(rf, 'C:/RPM/models/rf_classifier.pkl')