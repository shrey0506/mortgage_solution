from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    response: str

class ChatModel:
    def generate_response(self, user_input: str) -> str:
        return f"{user_input}"

class InputData(BaseModel):
    int64_field_0: int
    title: str
    first_name: str
    last_name: str
    date_of_birth: str
    marital_status: str
    nationality: str
    gender: str
    current_address: str
    residency_status: str
    mobile_number: str
    email_address: str
    number_of_jobs: int
    employment_status: str
    occupation: str
    contract_type: str
    company_name: str
    start_date: str
    basic_yearly_income: str
    additional_income: str
    expected_retirement_age: int
    number_of_child_dependents: int
    number_of_adult_dependents: int
    is_property_flat_or_leasehold_house: bool
    credit_score: int
    property_value: str
    deposit: str
    deposit_percentage: float
    loan_to_value: float
    total_borrowing_amount: str
    who_is_applying: str
    property_area: str
    area_risk_score: str
    job_risk_score: str
    company_risk_score: str
    base_rate: float
    repayment_type: str
    repayment_risk_score: str
    job_risk_score_float: float
    geo_spatial_risk_score: float
    income_to_loan_ratio: float
    approval_reason: str
    interest_rate_band: str
    passport_details: str
    social_security_number: str
    transunion_data: str