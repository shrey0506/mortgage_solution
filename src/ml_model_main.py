import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMClassifier

from mortgage.src.utils.bq_conn import bigquery_client
from mortgage.src.model.ml_model_training import ClassificationPipeline, RegressionPipeline


def run_model():
    # Step 1: Load data from BigQuery
    input_data = bigquery_client()
    print("Data loaded from BigQuery.")
    print(f"Initial shape: {input_data.shape}")

    # Step 2: Inspect target columns
    print("Sample target values before cleaning:")
    print(input_data[['mortgage_approved', 'interest_rate']].head())

    # Convert interest_rate to numeric, coerce invalid entries
    input_data['interest_rate'] = pd.to_numeric(input_data['interest_rate'], errors='coerce')

    # Map mortgage_approved from Yes/No or other formats to 1/0
    if input_data['mortgage_approved'].dtype == object:
        input_data['mortgage_approved'] = input_data['mortgage_approved'].map({'Yes': 1, 'No': 0, 'yes':1, 'no':0})

    # Drop rows with missing target values
    input_data = input_data.dropna(subset=['interest_rate', 'mortgage_approved'])
    print(f"Rows after dropping missing targets: {input_data.shape[0]}")

    if input_data.empty:
        raise ValueError("No data left after cleaning. Check data quality in BigQuery.")

    # Step 3: Split features and targets
    y_cls = input_data['mortgage_approved']
    y_reg = input_data['interest_rate']
    X = input_data.drop(columns=['mortgage_approved', 'interest_rate'])

    # Step 4: Identify column types
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

    # Remove problematic columns (e.g., nested JSON fields)
    for col in ['passport_details', 'transunion_data']:
        if col in categorical_cols:
            categorical_cols.remove(col)

    # Step 5: Build preprocessing pipeline
    full_preprocessor_pipeline = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # Step 6: Train-test split
    X_train, X_test, y_train_cls, y_test_cls = train_test_split(X, y_cls, test_size=0.2, random_state=42)
    _, _, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # Step 7: Train classification pipeline
    classification_model = LGBMClassifier(random_state=42)
    classification_pipeline = ClassificationPipeline(model=classification_model, preprocessor=full_preprocessor_pipeline)
    classification_pipeline.fit(X_train, y_train_cls)
    classification_pipeline.evaluate(X_test, y_test_cls)

    # Step 8: Train regression pipeline
    regression_model = RandomForestRegressor(random_state=42)
    regression_pipeline = RegressionPipeline(model=regression_model, preprocessor=full_preprocessor_pipeline)
    regression_pipeline.fit(X_train, y_train_reg)
    regression_pipeline.evaluate(X_test, y_test_reg)

    # Step 9: Save trained models
    base_path = os.path.join('mortgage', 'src', 'model')
    os.makedirs(base_path, exist_ok=True)

    with open(os.path.join(base_path, 'classification_pipeline.pkl'), 'wb') as f_cls:
        pickle.dump(classification_pipeline, f_cls)

    with open(os.path.join(base_path, 'regression_pipeline.pkl'), 'wb') as f_reg:
        pickle.dump(regression_pipeline, f_reg)

    # Step 10: Predict on entire dataset (optional)
    y_pred_cls = classification_pipeline.predict(X)
    y_pred_reg = regression_pipeline.predict(X)

    print("Model training and prediction complete.")

    return y_pred_cl
