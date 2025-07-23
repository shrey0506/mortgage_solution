import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime

fake = Faker()

# Constants
NUM_ROWS = 1000000
TITLES = ['Mr', 'Mrs', 'Ms', 'Dr']
MARITAL_STATUSES = ['Single', 'Married', 'Divorced', 'Widowed']
NATIONALITIES = ['UK', 'USA', 'India', 'Australia', 'Canada']
GENDERS = ['Male', 'Female', 'Other']
RESIDENCY_STATUSES = ['Owner', 'Renter', 'Family / friends']
EMPLOYMENT_STATUSES = ['Employed', 'Unemployed', 'Self-employed']
OCCUPATIONS = ['Technician', 'Manager', 'Clerk', 'Professional']
CONTRACT_TYPES = ['Permanent', 'Temporary', 'Contract']
COMPANIES = ['XYZ Ltd', 'ABC DEF', 'TechCorp', 'Innovate Inc']
YES_NO = ['Yes', 'No']
AREAS = ['London', 'Manchester', 'Birmingham', 'Glasgow', 'Leeds', 'Bristol']
REPAYMENT_TYPES = ['Capital & Interest', 'Interest Only']
REPAYMENT_RISK_SCORES = ['Good', 'Low', 'Moderate', 'High', 'Very High']

# Risk mappings
AREA_RISK_MAP = {
    'London': 'High',
    'Manchester': 'Medium',
    'Birmingham': 'Medium',
    'Glasgow': 'Low',
    'Leeds': 'Low',
    'Bristol': 'Medium'
}
JOB_RISK_MAP = {
    'Technician': 'Medium',
    'Manager': 'Low',
    'Clerk': 'High',
    'Professional': 'Low'
}
COMPANY_RISK_MAP = {
    'XYZ Ltd': 'High',
    'ABC DEF': 'Medium',
    'TechCorp': 'Low',
    'Innovate Inc': 'Medium'
}
risk_adjustment = {
    'Low': -0.25,
    'Medium': 0.0,
    'High': 0.5
}

base_rate_by_year = {
    2025: 4.25, 2024: 5.00, 2023: 5.25, 2022: 1.75, 2021: 0.25,
    2020: 0.10, 2019: 0.75, 2018: 0.75, 2017: 0.50, 2016: 0.25,
    2015: 0.50, 2014: 0.50, 2013: 0.50, 2012: 0.50, 2011: 0.50,
    2010: 0.50, 2009: 0.50, 2008: 5.00, 2007: 5.75, 2006: 4.75
}

def get_base_rate(start_date_str):
    try:
        year = datetime.strptime(start_date_str, "%b-%y").year
        return base_rate_by_year.get(year, 3.5)
    except:
        return 3.5

def generate_row():
    title = random.choice(TITLES)
    if title == 'Mr':
        gender = 'Male'
        marital_status = random.choice(MARITAL_STATUSES)
    elif title == 'Mrs':
        gender = 'Female'
        marital_status = 'Married'
    elif title == 'Ms':
        gender = 'Female'
        marital_status = random.choice(['Single', 'Divorced'])
    else:
        gender = random.choice(['Male', 'Female', 'Other'])
        marital_status = random.choice(MARITAL_STATUSES)

    employment_status = random.choice(EMPLOYMENT_STATUSES)
    if employment_status == 'Unemployed':
        occupation = "None"
        contract_type = "None"
        company_name = "None"
        basic_income = random.randint(0, 5000)
        additional_income = random.randint(0, 5000)
        number_of_jobs = 0
    else:
        occupation = random.choice(OCCUPATIONS)
        contract_type = random.choice(CONTRACT_TYPES)
        company_name = random.choice(COMPANIES)
        number_of_jobs = random.randint(1, 2)
        # Tiered income based on occupation
        if occupation in ['Manager', 'Professional']:
            basic_income = random.randint(70_000, 200_000)
        elif occupation == 'Technician':
            basic_income = random.randint(40_000, 90_000)
        else: # Clerk
            basic_income = random.randint(20_000, 50_000)
        additional_income = random.randint(0, 50_000)

    total_income = basic_income + additional_income
    property_value = round(random.uniform(100_000, 5_000_000), 2)
    deposit = round(random.uniform(0.05, 0.9) * property_value, 2)
    deposit_percentage = round((deposit / property_value) * 100, 2)
    loan_to_value = round(100 - deposit_percentage, 2)
    total_borrowing_amount = round(property_value - deposit, 2)
    income_to_loan_ratio = round(total_income / total_borrowing_amount, 2)
    credit_score = random.randint(300, 850)

    start_date = fake.date_between(start_date='-30y', end_date='today').strftime("%b-%y")
    base_rate = get_base_rate(start_date)

    if employment_status == 'Unemployed':
        # More restrictive approval criteria
        approved = ('Yes' if credit_score > 700 and additional_income > 3000 and loan_to_value < 85 else 'No')
    else:
        # Original criteria for employed/self-employed
        approved = ('Yes' if credit_score > 600 and loan_to_value < 90 else 'No')

    area = random.choice(AREAS)
    repayment_type = random.choice(REPAYMENT_TYPES)
    repayment_risk_score = random.choice(REPAYMENT_RISK_SCORES)
    job_risk_score_float = round(random.uniform(0.0, 1.0), 2)
    geo_spatial_risk_score = round(random.uniform(0.0, 1.0), 2)

    if approved == 'Yes':
        area_risk = AREA_RISK_MAP[area]
        job_risk = JOB_RISK_MAP.get(occupation, 'High') if occupation != 'None' else 'High'
        company_risk = COMPANY_RISK_MAP.get(company_name, 'High') if company_name != 'None' else 'High'
        total_risk_adjustment = (
            risk_adjustment[area_risk] +
            risk_adjustment[job_risk] + risk_adjustment[company_risk]
        )
        interest_rate = round(base_rate + total_risk_adjustment, 2)
        approval_reason = "Meets credit and LTV criteria"
        interest_rate_band = (
            "Low" if interest_rate < 3.5 else
            "Medium" if interest_rate < 5.0 else
            "High"
        )
    else:
        interest_rate = None
        approval_reason = "Credit score or LTV too high"
        interest_rate_band = "N/A"

    return {
        "title": title,
        "first_name": fake.first_name(),
        "last_name": fake.last_name(),
        "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=75).strftime("%d/%m/%Y"),
        "marital_status": marital_status,
        "nationality": random.choice(NATIONALITIES),
        "gender": gender,
        "current_address": fake.address().replace("\n", ", "),
        "residency_status": random.choice(RESIDENCY_STATUSES),
        "mobile_number": fake.phone_number(),
        "email_address": fake.email(),
        "number_of_jobs": number_of_jobs,
        "employment_status": employment_status,
        "occupation": occupation,
        "contract_type": contract_type,
        "company_name": company_name,
        "start_date": start_date,
        "basic_yearly_income": f"£{basic_income:,}",
        "additional_income": f"£{additional_income:,}",
        "expected_retirement_age": random.randint(60, 70),
        "number_of_child_dependents": random.randint(0, 4),
        "number_of_adult_dependents": random.randint(0, 2),
        "is_property_flat_or_leasehold_house": random.choice(YES_NO),
        "credit_score": credit_score,
        "property_value": f"£{property_value:,.2f}",
        "deposit": f"£{deposit:,.2f}",
        "deposit_percentage": f"{deposit_percentage}%",
        "loan_to_value": f"{loan_to_value}%",
        "total_borrowing_amount": f"£{total_borrowing_amount:,.2f}",
        "who_is_applying": "Just me",
        "mortgage_approved": approved,
        "property_area": area,
        "area_risk_score": AREA_RISK_MAP[area],
        "job_risk_score": JOB_RISK_MAP.get(occupation, 'High'),
        "company_risk_score": COMPANY_RISK_MAP.get(company_name, 'High'),
        "base_rate": f"{base_rate}%",
        "interest_rate": f"{interest_rate}%" if interest_rate else "N/A",
        "repayment_type": repayment_type,
        "repayment_risk_score": repayment_risk_score,
        "job_risk_score_float": job_risk_score_float,
        "geo_spatial_risk_score": geo_spatial_risk_score,
        "income_to_loan_ratio": income_to_loan_ratio,
        "approval_reason": approval_reason,
        "interest_rate_band": interest_rate_band
    }

# Generate dataset
data = [generate_row() for _ in range(NUM_ROWS)]
df = pd.DataFrame(data)