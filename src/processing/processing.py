import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Dict, Any
import json

# Custom Transformer for Winsorization (Outlier Handling)
class Winsorizer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for applying Winsorization to specified numerical columns.
    """
    def __init__(self, columns: Optional[List[str]] = None, lower_percentile: float = 0.05, upper_percentile: float = 0.95):
        self.columns = columns
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bounds_: Dict[str, Optional[float]] = {}
        self.upper_bounds_: Dict[str, Optional[float]] = {}
        self._fitted_columns: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fits the Winsorizer by calculating the bounds for winsorization.
        """
        X_copy = X.copy()
        if self.columns is None:
            self.columns = X_copy.select_dtypes(include=np.number).columns.tolist()

        numeric_cols_to_fit = [col for col in self.columns if col in X_copy.columns and pd.api.types.is_numeric_dtype(X_copy[col])]

        for col in numeric_cols_to_fit:
            self.lower_bounds_[col] = X_copy[col].quantile(self.lower_percentile)
            self.upper_bounds_[col] = X_copy[col].quantile(self.upper_percentile)

        self._fitted_columns = numeric_cols_to_fit
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Applies Winsorization to the specified numerical columns.
        """
        X_copy = X.copy()

        if self._fitted_columns is None:
            raise RuntimeError("Winsorizer has not been fitted yet.")

        for col in self._fitted_columns:
             if col in X_copy.columns and pd.api.types.is_numeric_dtype(X_copy[col]):
                lower_bound = self.lower_bounds_[col]
                upper_bound = self.upper_bounds_[col]
                X_copy[col] = np.where(X_copy[col] < lower_bound, lower_bound, X_copy[col])
                X_copy[col] = np.where(X_copy[col] > upper_bound, upper_bound, X_copy[col])

        return X_copy.values # Return as NumPy array for compatibility with ColumnTransformer


class MortgagePreprocessingPipeline:
    """
    A production-grade pipeline for preprocessing mortgage application data.
    """
    def __init__(self, numerical_cols_with_outliers: List[str]):
        self.numerical_cols_with_outliers = numerical_cols_with_outliers
        self.preprocessor: Optional[ColumnTransformer] = None
        self._numerical_features_final: Optional[List[str]] = None
        self._categorical_features_nominal_final: Optional[List[str]] = None
        self._label_encoder: Optional[LabelEncoder] = None


    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fits the preprocessing pipeline to the training data.
        """
        X_cleaned = self._initial_cleaning_and_feature_engineering(X.copy())

        numerical_features = X_cleaned.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()

        ordinal_categorical_features = ['interest_rate_band']
        categorical_features_nominal = [col for col in categorical_features if col not in ordinal_categorical_features]

        # Apply Label Encoding to the ordinal feature 'interest_rate_band' before ColumnTransformer
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

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('winsorizer', Winsorizer(columns=[col for col in numerical_features if col in self.numerical_cols_with_outliers])),
            ('scaler', StandardScaler())
        ])

        categorical_transformer_nominal = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self._numerical_features_final),
                ('cat', categorical_transformer_nominal, self._categorical_features_nominal_final)],
            remainder='passthrough'
        )

        self.preprocessor.fit(X_cleaned)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Applies the fitted preprocessing pipeline to the data.
        """
        if self.preprocessor is None:
            raise RuntimeError("Pipeline has not been fitted yet.")

        X_cleaned = self._initial_cleaning_and_feature_engineering(X.copy())

        # Apply the same Label Encoding transformation if the column exists
        ordinal_categorical_features = ['interest_rate_band']
        if ordinal_categorical_features[0] in X_cleaned.columns and self._label_encoder:
             X_cleaned[ordinal_categorical_features[0]] = self._label_encoder.transform(X_cleaned[ordinal_categorical_features[0]])


        return self.preprocessor.transform(X_cleaned)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """
        Fits the pipeline and transforms the data in one step.
        """
        self.fit(X, y)
        return self.transform(X)


    def _initial_cleaning_and_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Performs initial cleaning and feature engineering steps outside the main ColumnTransformer.
        """
        # Convert currency columns to numeric
        currency_cols = ['basic_yearly_income', 'additional_income', 'property_value', 'deposit', 'total_borrowing_amount']
        for col in currency_cols:
            if col in X.columns and X[col].dtype == 'object':
                X[col] = X[col].astype(str).str.replace('£', '', regex=False).str.replace(',', '', regex=False)
                X[col] = pd.to_numeric(X[col], errors='coerce')

        # Convert percentage columns to numeric
        percentage_cols = ['deposit_percentage', 'loan_to_value']
        for col in percentage_cols:
             if col in X.columns and X[col].dtype == 'object':
                X[col] = X[col].astype(str).str.replace('%', '', regex=False)
                X[col] = pd.to_numeric(X[col], errors='coerce')

        # Convert date_of_birth to age
        if 'date_of_birth' in X.columns:
            X['date_of_birth'] = pd.to_datetime(X['date_of_birth'], format='%d/%m/%Y', errors='coerce')
            today = datetime.now().date() # Use datetime.now().date() for current date
            X['age'] = X['date_of_birth'].apply(lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day)) if pd.notnull(dob) else None)

        # Convert start_date to employment duration (in months)
        if 'start_date' in X.columns:
            X['start_date'] = pd.to_datetime(X['start_date'], format='%b-%y', errors='coerce')
            today = datetime.now().date() # Use datetime.now().date() for current date
            X['months_at_company'] = X['start_date'].apply(
                lambda start_date: (today.year - start_date.year) * 12 + (today.month - start_date.month) if pd.notnull(start_date) else None
            )


        # Calculate total income
        if 'basic_yearly_income' in X.columns and 'additional_income' in X.columns:
            X['total_income'] = X['basic_yearly_income'] + X['additional_income']

        # Calculate debt-to-income ratio
        if 'total_income' in X.columns and 'total_borrowing_amount' in X.columns:
            X['debt_to_income_ratio'] = np.where(
                (X['total_income'].notnull()) & (X['total_income'] != 0),
                X['total_borrowing_amount'] / X['total_income'],
                np.nan
            )
            X['debt_to_income_ratio'] = X['debt_to_income_ratio'].replace([np.inf, -np.inf], np.nan)

        # Extract features from passport_details and transunion_data
        if 'passport_details' in X.columns:
            X['passport_details_dict'] = X['passport_details'].apply(lambda x: json.loads(x) if isinstance(x, str) and x != 'Missing' else ({} if x == 'Missing' else x))
            X['passport_country_of_issue'] = X['passport_details_dict'].apply(
                lambda x: x.get('country_of_issue') if isinstance(x, dict) and 'country_of_issue' in x else 'Missing'
            )
            X = X.drop(columns=['passport_details_dict'])

        if 'transunion_data' in X.columns:
             X['transunion_data_dict'] = X['transunion_data'].apply(lambda x: json.loads(x) if isinstance(x, str) and x != 'Missing' else ({} if x == 'Missing' else x))
             X['transunion_credit_score_band'] = X['transunion_data_dict'].apply(
                lambda x: x.get('credit_score_band') if isinstance(x, dict) and 'credit_score_band' in x else 'Missing'
            )
             X['transunion_has_adverse_markers'] = X['transunion_data_dict'].apply(
                lambda x: 'Yes' if isinstance(x, dict) and x.get('adverse_markers') else 'No'
            )
             X = X.drop(columns=['transunion_data_dict'])


        # Handle missing social_security_number by filling with 'Unknown'
        if 'social_security_number' in X.columns:
            X['social_security_number'] = X['social_security_number'].fillna('Unknown')

        # Drop original columns that have been transformed or are identifiers/targets
        columns_to_drop = [
            'first_name', 'last_name', 'date_of_birth', 'current_address',
            'mobile_number', 'email_address', 'start_date', 'passport_details',
            'transunion_data', 'approval_reason'
        ]
        X = X.drop(columns=[col for col in columns_to_drop if col in X.columns])

        return X

# Example Usage:
# Assuming 'df' is your raw DataFrame
# Identify numerical columns that showed outliers during EDA
# numerical_cols_with_outliers = ['total_income', 'property_value', 'total_borrowing_amount', 'debt_to_income_ratio', 'basic_yearly_income', 'additional_income', 'deposit']
#
# # Instantiate the pipeline
# preprocessing_pipeline = MortgagePreprocessingPipeline(numerical_cols_with_outliers=numerical_cols_with_outliers)
#
# # Fit and transform the data
# X_processed = preprocessing_pipeline.fit_transform(df.drop(columns=['mortgage_approved', 'interest_rate'])) # Exclude target variables
#
# print("\nShape of preprocessed data:", X_processed.shape)