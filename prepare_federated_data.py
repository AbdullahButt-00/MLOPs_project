#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ------------------- Config -------------------
DATAPATH = "E Commerce Dataset.xlsx"
SHEET_NAME = "E Comm"
OUTPUT_FOLDER = "./federated_data"
CLIENTS = 3  # Number of simulated clients/nodes
BATCH_SIZE = 8  # Small for CPU
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ------------------- Helpers -------------------
def make_onehot_encoder():
    # use the right kwarg depending on sklearn version (sparse_output added in newer versions)
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

# ------------------- Load and clean -------------------
def load_and_clean(path, sheet_name="E Comm"):
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.loc[df['PreferredLoginDevice'] == 'Phone', 'PreferredLoginDevice'] = 'Mobile Phone'
    df.loc[df['PreferedOrderCat'] == 'Mobile', 'PreferedOrderCat'] = 'Mobile Phone'
    df.loc[df['PreferredPaymentMode'] == 'COD', 'PreferredPaymentMode'] = 'Cash on Delivery'
    df.loc[df['PreferredPaymentMode'] == 'CC', 'PreferredPaymentMode'] = 'Credit Card'
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
    df['Churn'] = df['Churn'].astype(int)
    return df

# ------------------- Preprocessing -------------------
def build_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', make_onehot_encoder())
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    return preprocessor

def preprocess_node(df_node, preprocessor):
    X = df_node.drop(columns=['Churn'])
    y = df_node['Churn'].values
    X_trans = preprocessor.transform(X)
    return X_trans, y

# ------------------- Convert to TFF dataset -------------------
def create_tf_dataset(X, y, feature_names=None, batch_size=BATCH_SIZE):
    # feature_names should match transformed columns; if None, create numeric names
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    df_features = pd.DataFrame(X, columns=feature_names)
    ds = tf.data.Dataset.from_tensor_slices((dict(df_features), y))
    return ds.batch(batch_size)

# ------------------- Main -------------------
if __name__ == "__main__":
    df = load_and_clean(DATAPATH, SHEET_NAME)
    print("Raw data shape:", df.shape)

    # Use full dataset to build a stable preprocessor (avoids first-client bias)
    X_full = df.drop(columns=['Churn'])
    y_full = df['Churn'].values

    numeric_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_full.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_full)  # fit on entire dataset

    # Try to extract feature names for transformed matrix (useful to reconstruct DataFrame later)
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        # fallback: if ColumnTransformer doesn't support get_feature_names_out, leave None
        feature_names = None

    # Save metadata (sklearn version, original columns, feature names if available)
    metadata = {
        "sklearn_version": sklearn.__version__,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_names": feature_names
    }
    with open(os.path.join(OUTPUT_FOLDER, "metadata.json"), "w") as mf:
        json.dump(metadata, mf)
    print("Saved metadata ->", os.path.join(OUTPUT_FOLDER, "metadata.json"))

    # Simulate clients by shuffling and splitting
    client_dfs = np.array_split(df.sample(frac=1, random_state=RANDOM_STATE), CLIENTS)

    # Preprocess & save each client dataset (transformed numpy arrays + feature names)
    for idx, client_df in enumerate(client_dfs):
        X_trans, y = preprocess_node(client_df, preprocessor=preprocessor)
        file_path = os.path.join(OUTPUT_FOLDER, f"client_{idx+1}_data.pkl")
        with open(file_path, "wb") as f:
            pickle.dump({
                'X': X_trans,
                'y': y,
                'feature_names': feature_names
            }, f)
        print(f"Saved client {idx+1} data -> {file_path}")

        # optional: create and save a small TF dataset example file (not pickling tf.data.Dataset)
        # if you need TF datasets directly, load the pickle and call create_tf_dataset(...) when building the training loop

    # Save the fitted preprocessor (note: require compatible sklearn when loading)
    preproc_path = os.path.join(OUTPUT_FOLDER, "preprocessor.pkl")
    with open(preproc_path, "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"Saved preprocessor -> {preproc_path}")
