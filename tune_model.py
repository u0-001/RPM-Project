from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

# Load data
X_train = np.load('C:/RPM/data/X_train_features.npy')
y_train = np.load('C:/RPM/data/y_train.npy')
X_val = np.load('C:/RPM/data/X_val_features.npy')
y_val = np.load('C:/RPM/data/y_val.npy')

# Tune Random Forest
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 20, None]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc_ovr')
grid_search.fit(X_train, y_train)

print("Best RF parameters:", grid_search.best_params_)
best_rf = grid_search.best_estimator_

# Evaluate best RF
y_pred = best_rf.predict(X_val)
y_pred_proba = best_rf.predict_proba(X_val)
auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
cm = confusion_matrix(y_val, y_pred)
fpr = cm.sum(axis=0) - np.diag(cm)
total = cm.sum(axis=0)
fpr_rate = np.mean(fpr / total) * 100

print("Tuned AUROC:", auc)
print("Tuned FPR (%):", fpr_rate)

# Save tuned RF
import joblib
joblib.dump(best_rf, 'C:/RPM/models/rf_classifier_tuned.pkl')