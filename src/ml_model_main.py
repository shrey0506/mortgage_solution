import pandas as pd
import numpy as np
from typing import Union, Dict, Any

# Assuming MortgagePreprocessingPipeline class and a fitted classification model are available in the environment
# from your previous steps.
# Example:
# preprocessing_pipeline = MortgagePreprocessingPipeline(numerical_cols_with_outliers=...)
# preprocessing_pipeline.fit(X_train, y_train_cls) # Fit the pipeline on training data
# trained_classification_model = ... # Your trained classification model (e.g., LGBMClassifier)


def predict_mortgage_approval(
    raw_input_data: Union[pd.DataFrame, Dict[str, Any]],
    preprocessing_pipeline: object, # Expecting a fitted preprocessing pipeline instance
    classification_model: object    # Expecting a fitted classification model instance
) -> str:
    """
    Takes raw input data, preprocesses it using the provided pipeline, and predicts
    mortgage approval status using the provided classification model.

    Args:
        raw_input_data (Union[pd.DataFrame, Dict[str, Any]]): Raw input data for one or
                                                              multiple applicants. Can be
                                                              a pandas DataFrame or a
                                                              dictionary (for single applicant).
        preprocessing_pipeline (object): A fitted instance of the preprocessing pipeline.
        classification_model (object): A fitted instance of the classification model.

    Returns:
        str: Prediction result ('Yes' or 'No') for a single applicant, or a string
             representation of predictions for multiple applicants.
    """
    try:
        # Convert input to DataFrame if it's a dictionary (assuming single applicant)
        if isinstance(raw_input_data, dict):
            # Ensure the dictionary can be converted to a DataFrame row
            # This might require adding a dummy index or ensuring all expected columns are present
            input_df = pd.DataFrame([raw_input_data])
        elif isinstance(raw_input_data, pd.DataFrame):
            input_df = raw_input_data.copy()
        else:
            return "Error: Input data must be a pandas DataFrame or a dictionary."

        # Apply the preprocessing pipeline
        # Note: The pipeline should be already fitted on training data
        X_processed = preprocessing_pipeline.transform(input_df)

        # Make prediction using the classification model
        # Assuming the model predicts 0 for 'No' and 1 for 'Yes' based on LabelEncoder
        prediction_encoded = classification_model.predict(X_processed)

        # Convert prediction back to original labels ('Yes' or 'No')
        # This requires access to the LabelEncoder fitted on the target variable
        # A robust approach would be to include the LabelEncoder in the pipeline or pass it.
        # For simplicity here, assuming 0->No, 1->Yes based on common LabelEncoder behavior.
        # You might need to adjust this based on your specific LabelEncoder mapping.
        prediction_label = 'Yes' if prediction_encoded[0] == 1 else 'No' # Assuming single prediction

        if input_df.shape[0] == 1:
             return f"Predicted Mortgage Approval: {prediction_label}"
        else:
             # For multiple rows, return a list or Series of predictions
             # You would need access to the LabelEncoder's inverse_transform for this
             # For now, return a simplified representation
             return f"Predictions (encoded): {prediction_encoded.tolist()}"


    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Example Usage (requires fitting the pipeline and training a model first):
# Assuming:
# - 'preprocessing_pipeline' is a fitted instance of MortgagePreprocessingPipeline
# - 'classification_model' is a fitted instance of a classification model (e.g., LGBMClassifier)
# - 'raw_new_data' is a dictionary or DataFrame containing new applicant data

# Example raw data (replace with actual data structure matching your training data)
# raw_new_data = {
#     'title': 'Mr', 'first_name': 'John', 'last_name': 'Doe', 'date_of_birth': '15/07/1990',
#     'marital_status': 'Married', 'nationality': 'UK', 'gender': 'Male',
#     'current_address': '123 High St, London', 'residency_status': 'Owner',
#     'mobile_number': '07700 123456', 'email_address': 'john.doe@example.com',
#     'number_of_jobs': 1, 'employment_status': 'Employed', 'occupation': 'Tech/Finance',
#     'contract_type': 'Permanent', 'company_name': 'TechCorp', 'start_date': 'Jan-18',
#     'basic_yearly_income': '£60,000', 'additional_income': '£5,000',
#     'expected_retirement_age': 68, 'number_of_child_dependents': 0,
#     'number_of_adult_dependents': 0, 'is_property_flat_or_leasehold_house': 'No',
#     'credit_score': 750, 'property_value': '£400,000.00', 'deposit': '£80,000.00',
#     'deposit_percentage': '20.0%', 'loan_to_value': '80.0%',
#     'total_borrowing_amount': '£320,000.00', 'who_is_applying': 'Just me',
#     'property_area': 'London', 'area_risk_score': 'High', 'job_risk_score': 'Medium',
#     'company_risk_score': 'Low', 'base_rate': '4.25%',
#     'interest_rate': 'N/A', # Interest rate is the regression target, might not be in raw input for prediction
#     'repayment_type': 'Capital & Interest', 'repayment_risk_score': 'Good',
#     'job_risk_score_float': 0.5, 'geo_spatial_risk_score': 0.4,
#     'income_to_loan_ratio': 0.15, 'approval_reason': 'N/A', 'interest_rate_band': 'N/A',
#     'passport_details': '{"passport_number": "...", "issue_date": "...", "expiry_date": "...", "country_of_issue": "UK"}',
#     'social_security_number': 'AB123456A',
#     'transunion_data': '{"credit_score_band": "Excellent", "adverse_markers": [], "credit_utilization_ratio": 0.3, "years_of_credit_history": 10}'
# }
#
# prediction_result = predict_mortgage_approval(raw_new_data, preprocessing_pipeline, classification_model)
# print(prediction_result)