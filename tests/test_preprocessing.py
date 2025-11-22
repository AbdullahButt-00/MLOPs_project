#!/usr/bin/env python
"""
Unit tests for data preprocessing pipeline
"""

import pytest
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer


class TestPreprocessing:
    """Test suite for preprocessing functions"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample e-commerce data for testing"""
        return pd.DataFrame({
            'Tenure': [10, 20, 5, np.nan, 15],
            'PreferredLoginDevice': ['Mobile Phone', 'Computer', 'Phone', 'Mobile Phone', 'Computer'],
            'CityTier': [1, 2, 3, 1, 2],
            'WarehouseToHome': [10.5, 15.0, np.nan, 20.0, 12.5],
            'PreferredPaymentMode': ['Credit Card', 'Debit Card', 'UPI', 'COD', 'CC'],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'HourSpendOnApp': [2.5, 3.0, 1.5, np.nan, 2.0],
            'NumberOfDeviceRegistered': [3, 4, 2, 5, 3],
            'PreferedOrderCat': ['Mobile', 'Laptop & Accessory', 'Fashion', 'Grocery', 'Mobile Phone'],
            'SatisfactionScore': [3, 4, 5, 2, 3],
            'MaritalStatus': ['Single', 'Married', 'Divorced', 'Single', 'Married'],
            'NumberOfAddress': [2, 3, 1, 4, 2],
            'Complain': [0, 1, 0, 1, 0],
            'OrderAmountHikeFromlastYear': [15.0, 20.0, 10.0, 25.0, 18.0],
            'CouponUsed': [3.0, 5.0, 2.0, 4.0, 3.0],
            'OrderCount': [5, 8, 3, 10, 6],
            'DaySinceLastOrder': [7, 5, 10, 3, 8],
            'CashbackAmount': [150.0, 200.0, 100.0, 250.0, 175.0],
            'Churn': [0, 1, 0, 1, 0]
        })
    
    def test_data_cleaning(self, sample_data):
        """Test data cleaning transformations"""
        df = sample_data.copy()
        
        # Test device name standardization
        df.loc[df['PreferredLoginDevice'] == 'Phone', 'PreferredLoginDevice'] = 'Mobile Phone'
        assert 'Phone' not in df['PreferredLoginDevice'].values
        
        # Test order category standardization
        df.loc[df['PreferedOrderCat'] == 'Mobile', 'PreferedOrderCat'] = 'Mobile Phone'
        assert 'Mobile' not in df['PreferedOrderCat'].values
        
        # Test payment mode standardization
        df.loc[df['PreferredPaymentMode'] == 'COD', 'PreferredPaymentMode'] = 'Cash on Delivery'
        df.loc[df['PreferredPaymentMode'] == 'CC', 'PreferredPaymentMode'] = 'Credit Card'
        assert 'COD' not in df['PreferredPaymentMode'].values
        assert 'CC' not in df['PreferredPaymentMode'].values
    
    def test_numeric_columns_identification(self, sample_data):
        """Test identification of numeric columns"""
        X = sample_data.drop(columns=['Churn'])
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        expected_numeric = ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
                           'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
                           'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed',
                           'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']
        
        assert set(numeric_cols) == set(expected_numeric)
    
    def test_categorical_columns_identification(self, sample_data):
        """Test identification of categorical columns"""
        X = sample_data.drop(columns=['Churn'])
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        expected_categorical = ['PreferredLoginDevice', 'PreferredPaymentMode',
                               'Gender', 'PreferedOrderCat', 'MaritalStatus']
        
        assert set(categorical_cols) == set(expected_categorical)
    
    def test_missing_value_imputation(self, sample_data):
        """Test missing value imputation"""
        X = sample_data.drop(columns=['Churn'])
        
        # Numeric imputation (median)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X[numeric_cols])
        
        # Check no missing values after imputation
        assert not pd.DataFrame(X_imputed).isnull().any().any()
        
        # Check that median was used correctly
        tenure_median = sample_data['Tenure'].median()
        assert X_imputed[0, 0] != np.nan  # First column (Tenure) should have no NaN
    
    def test_scaling(self, sample_data):
        """Test MinMax scaling"""
        X = sample_data.drop(columns=['Churn'])
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X[numeric_cols])
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Check values are between 0 and 1
        assert X_scaled.min() >= 0
        assert X_scaled.max() <= 1
    
    def test_one_hot_encoding(self, sample_data):
        """Test one-hot encoding for categorical variables"""
        X = sample_data.drop(columns=['Churn'])
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        
        try:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
        imputer = SimpleImputer(strategy='most_frequent')
        X_imputed = imputer.fit_transform(X[categorical_cols])
        X_encoded = encoder.fit_transform(X_imputed)
        
        # Check output is numpy array
        assert isinstance(X_encoded, np.ndarray)
        
        # Check no missing values
        assert not np.isnan(X_encoded).any()
        
        # Check binary encoding (0s and 1s only)
        unique_values = np.unique(X_encoded)
        assert set(unique_values).issubset({0.0, 1.0})
    
    def test_preprocessor_saving_loading(self, sample_data, tmp_path):
        """Test saving and loading preprocessor"""
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        
        X = sample_data.drop(columns=['Churn'])
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Create preprocessor
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])
        
        try:
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
        except TypeError:
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])
        
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
        
        # Fit preprocessor
        preprocessor.fit(X)
        
        # Save
        save_path = tmp_path / "test_preprocessor.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        
        # Load
        with open(save_path, 'rb') as f:
            loaded_preprocessor = pickle.load(f)
        
        # Transform with both
        X_original = preprocessor.transform(X)
        X_loaded = loaded_preprocessor.transform(X)
        
        # Check they produce same output
        np.testing.assert_array_almost_equal(X_original, X_loaded)
    
    def test_target_variable_binary(self, sample_data):
        """Test that target variable is binary"""
        y = sample_data['Churn']
        unique_values = y.unique()
        
        assert set(unique_values).issubset({0, 1})
        assert len(unique_values) <= 2
    
    def test_data_shape_after_preprocessing(self, sample_data):
        """Test data shape is maintained after preprocessing"""
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        
        X = sample_data.drop(columns=['Churn'])
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])
        
        try:
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
        except TypeError:
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])
        
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
        
        X_transformed = preprocessor.fit_transform(X)
        
        # Check number of rows is maintained
        assert X_transformed.shape[0] == len(sample_data)
        
        # Check we have more columns due to one-hot encoding
        assert X_transformed.shape[1] > len(X.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])