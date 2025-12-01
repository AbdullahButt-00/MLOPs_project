#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import json
import pandas as pd
import numpy as np
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
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

# ------------------- Load and clean -------------------
def load_and_clean(path, sheet_name="E Comm"):
    df = pd.read_excel(path, sheet_name=sheet_name)
    # canonicalize some values
    if 'PreferredLoginDevice' in df.columns:
        df.loc[df['PreferredLoginDevice'] == 'Phone', 'PreferredLoginDevice'] = 'Mobile Phone'
    if 'PreferedOrderCat' in df.columns:
        df.loc[df['PreferedOrderCat'] == 'Mobile', 'PreferedOrderCat'] = 'Mobile Phone'
    if 'PreferredPaymentMode' in df.columns:
        df.loc[df['PreferredPaymentMode'] == 'COD', 'PreferredPaymentMode'] = 'Cash on Delivery'
        df.loc[df['PreferredPaymentMode'] == 'CC', 'PreferredPaymentMode'] = 'Credit Card'
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
    if 'Churn' in df.columns:
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
    ], remainder='drop')
    return preprocessor

def get_transformed_feature_names(preprocessor, numeric_cols, categorical_cols):
    # numeric names are the same as numeric_cols
    names = []
    # ColumnTransformer stores transformers_ after fit
    for name, trans, cols in preprocessor.transformers_:
        if name == 'num':
            names.extend(cols)
        elif name == 'cat':
            # `trans` is a Pipeline: ('imputer', SimpleImputer), ('ohe', OneHotEncoder)
            # find OHE and get categories
            ohe = None
            if hasattr(trans, 'named_steps') and 'ohe' in trans.named_steps:
                ohe = trans.named_steps['ohe']
            elif hasattr(trans, 'steps') and len(trans.steps) > 0:
                # fallback
                for step_name, step in trans.steps:
                    if hasattr(step, 'get_feature_names_out'):
                        ohe = step
                        break
            if ohe is not None:
                try:
                    # sklearn >=1.0
                    cat_names = ohe.get_feature_names_out(cols).tolist()
                except Exception:
                    # fallback older versions: build manually from categories_
                    cat_names = []
                    if hasattr(ohe, 'categories_'):
                        for col, cats in zip(cols, ohe.categories_):
                            for cat in cats:
                                cat_names.append(f"{col}__{cat}")
                    else:
                        # final fallback: column name only
                        cat_names = cols
                names.extend(cat_names)
            else:
                # if we can't find ohe, add original categorical names
                names.extend(cols)
    return names

def preprocess_node(df_node, preprocessor):
    X = df_node.drop(columns=['Churn'])
    y = df_node['Churn'].values
    X_trans = preprocessor.transform(X)
    return X_trans, y

# ------------------- Main -------------------
if __name__ == "__main__":
    df = load_and_clean(DATAPATH, SHEET_NAME)
    print("Raw data shape:", df.shape)

    if 'Churn' not in df.columns:
        raise RuntimeError("Churn column not found in dataset.")

    # Use full dataset to build a stable preprocessor (avoids first-client bias)
    X_full = df.drop(columns=['Churn'])
    y_full = df['Churn'].values

    numeric_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_full.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_full)  # fit on entire dataset

    # derive transformed feature names (robust)
    try:
        transformed_feature_names = get_transformed_feature_names(preprocessor, numeric_cols, categorical_cols)
    except Exception:
        transformed_feature_names = None

    # Save metadata (sklearn version, original columns, feature names if available)
    metadata = {
        "sklearn_version": sklearn.__version__,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "transformed_feature_names": transformed_feature_names
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
                'transformed_feature_names': transformed_feature_names
            }, f)
        print(f"Saved client {idx+1} data -> {file_path}")

    # Save the fitted preprocessor (note: require compatible sklearn when loading)
    preproc_path = os.path.join(OUTPUT_FOLDER, "preprocessor.pkl")
    with open(preproc_path, "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"Saved preprocessor -> {preproc_path}")
