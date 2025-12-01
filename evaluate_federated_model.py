#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils import shuffle

# ------------------- Config -------------------
DATAPATH = "E Commerce Dataset.xlsx"
SHEET_NAME = "E Comm"
DATA_FOLDER = "federated_data"
MODEL_PATH = os.path.join(DATA_FOLDER, "federated_churn_model.h5")
PREPROCESSOR_PATH = os.path.join(DATA_FOLDER, "preprocessor.pkl")
METADATA_PATH = os.path.join(DATA_FOLDER, "metadata.json")
BATCH_SIZE = 8
RANDOM_STATE = 42

# ------------------- Load preprocessor and model -------------------
with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

transformed_feature_names = metadata.get("transformed_feature_names", None)

model = tf.keras.models.load_model(MODEL_PATH)
print("Loaded federated model and preprocessor")

# ------------------- Load test dataset -------------------
df = pd.read_excel(DATAPATH, sheet_name=SHEET_NAME)
df['Churn'] = df['Churn'].astype(int)
df = shuffle(df, random_state=RANDOM_STATE)

test_df = df.sample(frac=0.2, random_state=RANDOM_STATE)
X_test = test_df.drop(columns=['Churn'])
y_test = test_df['Churn'].values

# ------------------- Transform features -------------------
X_test_trans = preprocessor.transform(X_test).astype(np.float32)

# ------------------- Predict and evaluate -------------------
y_pred_prob = model.predict(X_test_trans, batch_size=BATCH_SIZE).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print("\nEvaluation metrics on test set:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------- SAVE METRICS TO FOLDER ----------------
OUTPUT_DIR = os.path.join(DATA_FOLDER, "evaluation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

metric_text = (
    f"Accuracy: {acc:.4f}\n"
    f"Precision: {prec:.4f}\n"
    f"Recall: {rec:.4f}\n"
    f"F1-score: {f1:.4f}\n"
    f"ROC AUC: {roc_auc:.4f}\n\n"
    "Classification Report:\n"
    f"{classification_report(y_test, y_pred, digits=4)}\n"
    "Confusion Matrix:\n"
    f"{confusion_matrix(y_test, y_pred)}\n"
)

with open(os.path.join(OUTPUT_DIR, "evaluation_metrics.txt"), "w") as f:
    f.write(metric_text)

# Save confusion matrix separately
cm = confusion_matrix(y_test, y_pred)
np.savetxt(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"), cm, delimiter=",")

print(f"\nSaved evaluation results to folder: {OUTPUT_DIR}")

# ------------------- Churn probability distribution (aesthetic) -------------------
plt.style.use("seaborn-darkgrid")
plt.figure(figsize=(8,5))
plt.hist(y_pred_prob, bins=20, edgecolor='black', alpha=0.85)
plt.title("Predicted Churn Probability Distribution", fontsize=14)
plt.xlabel("Predicted probability", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "probability_distribution.png"), dpi=160)
plt.savefig(os.path.join(OUTPUT_DIR, "probability_distribution.svg"))
plt.close()

# ------------------- Approximate feature importance (permutation) -------------------
print("\nCalculating approximate feature importance (permutation)...")
baseline_acc = accuracy_score(y_test, y_pred)

n_features = X_test_trans.shape[1]
if transformed_feature_names and len(transformed_feature_names) == n_features:
    feature_names = transformed_feature_names
else:
    feature_names = [f"f{i}" for i in range(n_features)]

importances = []

for i in range(n_features):
    X_permuted = X_test_trans.copy()
    np.random.shuffle(X_permuted[:, i])
    y_pred_perm_prob = model.predict(X_permuted, batch_size=BATCH_SIZE).flatten()
    y_pred_perm = (y_pred_perm_prob >= 0.5).astype(int)
    perm_acc = accuracy_score(y_test, y_pred_perm)
    importances.append(baseline_acc - perm_acc)

# Plot top features
fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
fi_df = fi_df.sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10,6))
plt.barh(fi_df['feature'][::-1], fi_df['importance'][::-1], alpha=0.9)
plt.xlabel("Accuracy decrease (permutation importance)", fontsize=12)
plt.title("Top feature importances (approximate)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=160)
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.svg"))
plt.close()

print("\nAll plots and metrics saved successfully.")
