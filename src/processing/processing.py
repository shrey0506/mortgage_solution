import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Step 1: Currency Cleanup
def clean_currency(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['property_value', 'deposit', 'total_borrowing_amount', 'deposit_percentage',
            'loan_to_value', 'additional_income', 'base_rate', 'interest_rate', 'basic_yearly_income']
    for col in cols:
        df[col] = df[col].astype(str).str.replace('€', '', regex=True).str.replace(',', '', regex=True).str.replace('%', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Step 2: Combine dependents
def combine_dependents(df: pd.DataFrame) -> pd.DataFrame:
    df['number_of_dependents'] = df['number_of_child_dependents'].astype(int) + df['number_of_adult_dependents'].astype(int)
    return df

# Step 3: Categorical Label Encoding
def encode_basic_labels(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['nationality', 'gender', 'residency_status', 'who_is_applying', 'title',
            'employment_status', 'marital_status', 'occupation', 'contract_type',
            'is_property_flat_or_leasehold_house', 'mortgage_approved', 'property_area',
            'area_risk_score', 'job_risk_score', 'company_risk_score']
    le = LabelEncoder()
    for col in cols:
        df[col] = df[col].astype(str)
        df[f"{col}_encoded"] = le.fit_transform(df[col])
    return df

def generate_approval_reason(row: pd.Series) -> str:
    reasons = []
    if row['credit_score'] >= 600 and row['loan_to_value'] <= 75:
        reasons.append("Approved due to high credit score and low loan-to-value ratio.")
    if row['income_to_loan_ratio'] >= 3:
        reasons.append("Approved due to strong income-to-loan ratio.")
    if row['deposit_percentage'] >= 20:
        reasons.append("Approved due to large deposit.")
    if row['job_risk_score'] <= 3:
        reasons.append("Approved due to low job risk.")
    if row['repayment_risk_score'] <= 3:
        reasons.append("Approved due to low repayment risk.")
    if row['area_risk_score'] <= 3:
        reasons.append("Approved due to low area risk.")
    if row['company_risk_score'] <= 3:
        reasons.append("Approved due to stable employer.")
    if row['who_is_applying'] == 'Joint':
        reasons.append("Approved due to joint application with shared financial responsibility.")
    if row['credit_score'] < 600:
        reasons.append("Rejected due to low credit score.")
    if row['loan_to_value'] > 90:
        reasons.append("Rejected due to high loan-to-value ratio.")
    if row['income_to_loan_ratio'] < 2:
        reasons.append("Rejected due to insufficient income relative to loan.")
    if row['job_risk_score'] > 7:
        reasons.append("Rejected due to high job risk.")
    if row['repayment_risk_score'] > 7:
        reasons.append("Rejected due to high repayment risk.")
    if row['area_risk_score'] > 7:
        reasons.append("Rejected due to high area risk.")
    if row['deposit_percentage'] < 10:
        reasons.append("Rejected due to low deposit.")
    if row['company_risk_score'] > 7:
        reasons.append("Rejected due to high company risk.")
    if row.get('interest_rate_band') == "High":
        reasons.append("Further review required due to high interest rate band.")

    if not reasons:
        return "Further review required."
    return ". ".join(reasons)

# Step 5: Feature Engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce').fillna(pd.Timestamp('1978-01-01'))
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce').fillna(pd.Timestamp('2000-01-01'))
    df['age'] = datetime.now().year - df['date_of_birth'].dt.year
    df['employment_length_years'] = (datetime.now() - df['start_date']).dt.days // 365
    df['total_income'] = df['basic_yearly_income'] + df['additional_income']
    df['total_dependents'] = df['number_of_child_dependents'] + df['number_of_adult_dependents']
    df['dependents_to_income_ratio'] = df['total_dependents'] / (df['total_income'] + 1e-6) # Add small epsilon to avoid division by zero
    df['location_risk_score'] = df['area_risk_score'] + df['geo_spatial_risk_score']
    df['income_stability_score'] = df['employment_length_years'] / (df['job_risk_score'] + df['company_risk_score'] + 1)
    df['interest_spread'] = df['interest_rate'] - df['base_rate']
    df['monthly_repayment_estimate'] = (df['total_borrowing_amount'] * df['interest_rate'] / 100) / 12
    df['affordability_score'] = df['income_to_loan_ratio'] - df['repayment_risk_score']
    df['is_married'] = df['marital_status'].apply(lambda x: 1 if str(x).strip().lower() == 'married' else 0)
    df['is_female'] = df['gender'].apply(lambda x: 1 if str(x).strip().lower() == 'female' else 0)
    df['is_joint_application'] = df['who_is_applying'].apply(lambda x: 1 if str(x).strip().lower() == 'joint' else 0)
    df['is_leasehold'] = df['is_property_flat_or_leasehold_house'].apply(lambda x: 1 if 'leasehold' in str(x).lower() else 0)
    df['approval_reason'] = df.apply(generate_approval_reason, axis=1)
    df['approval_reason_encoded'] = LabelEncoder().fit_transform(df['approval_reason'].astype(str))
    df['age_left_for_retirement'] = df['expected_retirement_age'] - df['age']
    # Random categorical risk assignments - This part seems incorrect, will keep original values
    # risk_categories = ['Very High', 'High', 'Moderate', 'Low', 'no risk']
    # for col in ['area_risk_score', 'location_risk_score', 'job_risk_score', 'company_risk_score', 'repayment_risk_score', 'income_stability_score']:
    #     df[col] = np.random.choice(risk_categories, size=df.shape[0])
    #     df[col] = df[col].map({"no risk":1, "Low":2, "Moderate":3, "High":4, "Very High":5})
    return df

# Step 6: Drop Leakage & Useless Columns
def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ['interest_rate', 'interest_rate_band', 'interest_spread', 'monthly_repayment_estimate',
                 'approval_reason', 'approval_reason_encoded','affordability_score', 'who_is_applying',
                 'who_is_applying_encoded', 'age_group', 'start_date', 'date_of_birth', 'current_address',
                 'mobile_number', 'email_address', 'company_name', 'property_area']
    return df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# Step 7: Encode ALL Remaining Objects
def encode_remaining_objects(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

# Step 8: Scale Important Numerical Columns
def scale_numerics(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    cols = ['property_value', 'total_income', 'total_borrowing_amount', 'deposit'] # Corrected column names
    # Ensure columns exist before scaling
    cols_to_scale = [col for col in cols if col in df.columns]
    if cols_to_scale:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df


def run_pipeline(filepath: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(filepath, low_memory=False)
    pipeline = Pipeline([
        ('currency_cleaning', FunctionTransformer(clean_currency)),
        ('combine_dependents', FunctionTransformer(combine_dependents)),
        ('encode_basic_labels', FunctionTransformer(encode_basic_labels)),
        ('engineer_features', FunctionTransformer(engineer_features)),
        ('drop_leakage_columns', FunctionTransformer(drop_leakage)),
        ('encode_remaining_objects', FunctionTransformer(encode_remaining_objects)),
        ('scale_numeric_features', FunctionTransformer(scale_numerics))
    ])
    df_processed = pipeline.fit_transform(df)

    # Prepare for modeling
    target = 'mortgage_approved_encoded' # Use encoded target
    X = df_processed.drop(columns=[target] if target in df_processed.columns else [])
    y = df_processed[target] if target in df_processed.columns else pd.Series()

    if not y.empty:
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        # Feature selection using RFE
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe = RFE(estimator=clf, n_features_to_select=15)
        rfe.fit(X_train, y_train)
        selected_features = X_train.columns[rfe.support_].tolist()
        return df_processed, selected_features, X_train, y_train
    else:
        print(f"Target column '{target}' not found in the processed DataFrame.")
        return df_processed, [], pd.DataFrame(), pd.Series()