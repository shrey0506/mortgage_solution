import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Dict, Any
from datetime import datetime
import json


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Optional[List[int]] = None, lower_percentile: float = 0.05, upper_percentile: float = 0.95):
        self.columns = columns
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bounds_: Dict[int, float] = {}
        self.upper_bounds_: Dict[int, float] = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        selected_indices = self.columns or range(X_df.shape[1])
        for idx in selected_indices:
            col_data = X_df.iloc[:, idx]
            self.lower_bounds_[idx] = col_data.quantile(self.lower_percentile)
            self.upper_bounds_[idx] = col_data.quantile(self.upper_percentile)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        selected_indices = self.columns or range(X_df.shape[1])
        for idx in selected_indices:
            lower = self.lower_bounds_[idx]
            upper = self.upper_bounds_[idx]
            X_df.iloc[:, idx] = np.clip(X_df.iloc[:, idx], lower, upper)
        return X_df.values


class MortgagePreprocessingPipeline:
    def __init__(self, numerical_cols_with_outliers: List[str]):
        self.numerical_cols_with_outliers = numerical_cols_with_outliers
        self.preprocessor: Optional[ColumnTransformer] = None
        self._numerical_features_final: Optional[List[str]] = None
        self._categorical_features_nominal_final: Optional[List[str]] = None
        self._label_encoder: Optional[LabelEncoder] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X_cleaned = self._initial_cleaning_and_feature_engineering(X.copy())

        numerical_features = X_cleaned.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()

        ordinal_categorical_features = ['interest_rate_band']
        categorical_features_nominal = [col for col in categorical_features if col not in ordinal_categorical_features]

        if ordinal_categorical_features[0] in X_cleaned.columns:
            self._label_encoder = LabelEncoder()
            X_cleaned[ordinal_categorical_features[0]] = self._label_encoder.fit_transform(X_cleaned[ordinal_categorical_features[0]])
            numerical_features_final = numerical_features + ordinal_categorical_features
            categorical_features_nominal_final = [col for col in categorical_features_nominal if col not in ordinal_categorical_features]
        else:
            numerical_features_final = numerical_features
            categorical_features_nominal_final = categorical_features_nominal

        self._numerical_features_final = numerical_features_final
        self._categorical_features_nominal_final = categorical_features_nominal_final

        # Convert outlier column names to indices
        winsorizer_column_indices = [
            i for i, col in enumerate(numerical_features_final)
            if col in self.numerical_cols_with_outliers
        ]

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('winsorizer', Winsorizer(columns=winsorizer_column_indices)),
            ('scaler', StandardScaler())
        ])

        categorical_transformer_nominal = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self._numerical_features_final),
                ('cat', categorical_transformer_nominal, self._categorical_features_nominal_final)
            ],
            remainder='passthrough'
        )

        self.preprocessor.fit(X_cleaned)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise RuntimeError("Pipeline has not been fitted yet.")

        X_cleaned = self._initial_cleaning_and_feature_engineering(X.copy())

        if 'interest_rate_band' in X_cleaned.columns and self._label_encoder:
            X_cleaned['interest_rate_band'] = self._label_encoder.transform(X_cleaned['interest_rate_band'])

        return self.preprocessor.transform(X_cleaned)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _initial_cleaning_and_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        currency_cols = ['basic_yearly_income', 'additional_income', 'property_value', 'deposit', 'total_borrowing_amount']
        for col in currency_cols:
            if col in X.columns and X[col].dtype == 'object':
                X[col] = X[col].astype(str).str.replace('£', '', regex=False).str.replace(',', '', regex=False)
                X[col] = pd.to_numeric(X[col], errors='coerce')

        percentage_cols = ['deposit_percentage', 'loan_to_value']  # fixed typo from deposit_pāercentage
        for col in percentage_cols:
            if col in X.columns and X[col].dtype == 'object':
                X[col] = X[col].astype(str).str.replace('%', '', regex=False)
                X[col] = pd.to_numeric(X[col], errors='coerce')

        if 'date_of_birth' in X.columns:
            X['date_of_birth'] = pd.to_datetime(X['date_of_birth'], format='%d/%m/%Y', errors='coerce')
            today = datetime.now().date()
            X['age'] = X['date_of_birth'].apply(
                lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day)) if pd.notnull(dob) else None
            )

        if 'start_date' in X.columns:
            X['start_date'] = pd.to_datetime(X['start_date'], format='%b-%y', errors='coerce')
            today = datetime.now().date()
            X['months_at_company'] = X['start_date'].apply(
                lambda d: (today.year - d.year) * 12 + (today.month - d.month) if pd.notnull(d) else None
            )

        if 'basic_yearly_income' in X.columns and 'additional_income' in X.columns:
            X['total_income'] = X['basic_yearly_income'] + X['additional_income']

        if 'total_income' in X.columns and 'total_borrowing_amount' in X.columns:
            X['debt_to_income_ratio'] = np.where(
                (X['total_income'].notnull()) & (X['total_income'] != 0),
                X['total_borrowing_amount'] / X['total_income'],
                np.nan
            )
            X['debt_to_income_ratio'] = X['debt_to_income_ratio'].replace([np.inf, -np.inf], np.nan)

        if 'passport_details' in X.columns:
            X['passport_details_dict'] = X['passport_details'].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x != 'Missing' else {}
            )
            X['passport_country_of_issue'] = X['passport_details_dict'].apply(
                lambda x: x.get('country_of_issue') if isinstance(x, dict) else 'Missing'
            )
            X = X.drop(columns=['passport_details_dict'])

        if 'transunion_data' in X.columns:
            X['transunion_data_dict'] = X['transunion_data'].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x != 'Missing' else {}
            )
            X['transunion_credit_score_band'] = X['transunion_data_dict'].apply(
                lambda x: x.get('credit_score_band') if isinstance(x, dict) else 'Missing'
            )
            X['transunion_has_adverse_markers'] = X['transunion_data_dict'].apply(
                lambda x: 'Yes' if isinstance(x, dict) and x.get('adverse_markers') else 'No'
            )
            X = X.drop(columns=['transunion_data_dict'])

        if 'social_security_number' in X.columns:
            X['social_security_number'] = X['social_security_number'].fillna('Unknown')

        columns_to_drop = [
            'first_name', 'last_name', 'date_of_birth', 'current_address',
            'mobile_number', 'email_address', 'start_date', 'passport_details',
            'transunion_data', 'approval_reason'
        ]
        X = X.drop(columns=[col for col in columns_to_drop if col in X.columns])

        return X
