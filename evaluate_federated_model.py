#!/usr/bin/env python
# coding: utf-8

import os
import pickle
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
BATCH_SIZE = 8
RANDOM_STATE = 42

# ------------------- Load preprocessor and model -------------------
with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

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
X_test_trans = preprocessor.transform(X_test)

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

# ------------------- Churn probability distribution -------------------
plt.figure(figsize=(6,4))
plt.hist(y_pred_prob, bins=20, color='skyblue', edgecolor='black')
plt.title("Predicted Churn Probability Distribution")
plt.xlabel("Predicted probability")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ------------------- Approximate feature importance -------------------
# Permutation-based importance (CPU-friendly)
print("\nCalculating approximate feature importance (permutation)...")
baseline_acc = accuracy_score(y_test, y_pred)

feature_names = X_test.columns.tolist()
importances = []

for i, col in enumerate(feature_names):
    X_permuted = X_test_trans.copy()
    np.random.shuffle(X_permuted[:, i])  # shuffle one column
    y_pred_perm = (model.predict(X_permuted, batch_size=BATCH_SIZE).flatten() >= 0.5).astype(int)
    perm_acc = accuracy_score(y_test, y_pred_perm)
    importances.append(baseline_acc - perm_acc)

# Plot top features
fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
fi_df = fi_df.sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(8,6))
plt.barh(fi_df['feature'][::-1], fi_df['importance'][::-1], color='orange')
plt.xlabel("Accuracy decrease (permutation importance)")
plt.title("Top feature importances (approximate)")
plt.tight_layout()
plt.show()
