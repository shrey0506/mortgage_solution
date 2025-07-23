import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime
import re # Import regex for NI number validation
import json # Import json for serializing dicts

fake = Faker()

# Constants
NUM_ROWS = 1000
TITLES = ['Mr', 'Mrs', 'Ms', 'Dr']
MARITAL_STATUSES = ['Single', 'Married', 'Divorced', 'Widowed']
NATIONALITIES = ['UK', 'USA', 'India', 'Australia', 'Canada'] # Consider adding more common UK nationalities
GENDERS = ['Male', 'Female', 'Other']
RESIDENCY_STATUSES = ['Owner', 'Renter', 'Family / friends']
EMPLOYMENT_STATUSES = ['Employed', 'Unemployed', 'Self-employed', 'Retired', 'Student'] # Added Retired and Student
OCCUPATIONS = ['NHS/Healthcare', 'Education/Public', 'Tech/Finance', 'Construction/Transport', 'Gig/Freelancers', 'Retail', 'Hospitality', 'Manufacturing', 'Other'] # More UK-specific sectors
CONTRACT_TYPES = ['Permanent', 'Temporary', 'Contract', 'Zero-hour', 'Fixed-term'] # Added Zero-hour and Fixed-term
COMPANIES = ['NHS Trust', 'Local Council', 'HSBC', 'Barclays', 'Tesco', 'Sainsbury\'s', 'Balfour Beatty', 'Laing O\'Rourke', 'Freelance', 'Small Business'] # More UK companies/types
YES_NO = ['Yes', 'No']
AREAS = ['London', 'Manchester', 'Birmingham', 'Glasgow', 'Leeds', 'Bristol', 'Newcastle', 'Liverpool', 'Sheffield', 'Cardiff', 'Edinburgh'] # Added more UK cities
REPAYMENT_TYPES = ['Capital & Interest', 'Interest Only']
REPAYMENT_RISK_SCORES = ['Good', 'Low', 'Moderate', 'High', 'Very High']

# Realistic UK income ranges (approximate)
INCOME_RANGES = {
    'NHS/Healthcare': (25000, 80000),
    'Education/Public': (22000, 70000),
    'Tech/Finance': (35000, 150000),
    'Construction/Transport': (28000, 60000),
    'Gig/Freelancers': (15000, 100000), # Highly variable
    'Retail': (18000, 40000),
    'Hospitality': (16000, 35000),
    'Manufacturing': (20000, 50000),
    'Other': (20000, 70000)
}

# Risk mappings (updated and expanded)
AREA_RISK_MAP = {
    'London': 'High',
    'Manchester': 'Medium',
    'Birmingham': 'Medium',
    'Glasgow': 'Low',
    'Leeds': 'Low',
    'Bristol': 'Medium',
    'Newcastle': 'Low',
    'Liverpool': 'Medium',
    'Sheffield': 'Low',
    'Cardiff': 'Medium',
    'Edinburgh': 'Medium'
}
JOB_RISK_MAP = {
    'NHS/Healthcare': 'Low',
    'Education/Public': 'Low-Medium',
    'Tech/Finance': 'Medium',
    'Construction/Transport': 'High',
    'Gig/Freelancers': 'Very High',
    'Retail': 'Medium',
    'Hospitality': 'High',
    'Manufacturing': 'Medium',
    'Other': 'Medium'
}
COMPANY_RISK_MAP = { # Simplified for example
    'NHS Trust': 'Low',
    'Local Council': 'Low',
    'HSBC': 'Low',
    'Barclays': 'Low',
    'Tesco': 'Medium',
    'Sainsbury\'s': 'Medium',
    'Balfour Beatty': 'High',
    'Laing O\'Rourke': 'High',
    'Freelance': 'Very High',
    'Small Business': 'High'
}
risk_adjustment = {
    'Low': -0.5,
    'Low-Medium': -0.25,
    'Medium': 0.0,
    'High': 0.5,
    'Very High': 1.0
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
        return base_rate_by_year.get(year, 3.5) # Defaulting to a reasonable rate if year is outside range
    except:
        return 3.5

# Function to generate a realistic-looking UK National Insurance Number
def generate_ni_number():
    prefix = random.choice(['AB', 'CD', 'EF', 'GH', 'JK', 'LM', 'NP', 'OR', 'ST', 'WZ']) # Valid prefixes
    digits = ''.join(random.choices('0123456789', k=6))
    suffix = random.choice(['A', 'B', 'C', 'D']) # Valid suffixes
    return f"{prefix}{digits}{suffix}"

# Function to simulate TransUnion data
def simulate_transunion_data(credit_score):
    data = {
        "credit_score_band": None,
        "adverse_markers": [], # e.g., CCJ, default, late payment
        "credit_utilization_ratio": round(random.uniform(0.1, 0.8), 2),
        "years_of_credit_history": random.randint(1, 30)
    }

    if credit_score >= 700:
        data["credit_score_band"] = "Excellent"
        if random.random() < 0.05: # Small chance of minor adverse
             data["adverse_markers"].append(random.choice(["Late Payment (60 days)", "Small Default"]))
    elif credit_score >= 600:
        data["credit_score_band"] = "Good"
        if random.random() < 0.2:
             data["adverse_markers"].append(random.choice(["Late Payment (60 days)", "Small Default", "CCJ (Satisfied)"]))
    elif credit_score >= 500:
        data["credit_score_band"] = "Fair"
        if random.random() < 0.4:
             data["adverse_markers"].append(random.choice(["Late Payment (90 days)", "Default", "CCJ (Unsatisfied)", "Missed Mortgage Payment"]))
    else:
        data["credit_score_band"] = "Poor"
        if random.random() < 0.7:
             data["adverse_markers"].append(random.choice(["Late Payment (90+ days)", "Significant Default", "CCJ (Unsatisfied)", "Bankruptcy", "IVA"]))

    return data

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
    occupation = "None"
    contract_type = "None"
    company_name = "None"
    basic_income = 0
    additional_income = 0
    number_of_jobs = 0

    if employment_status == 'Employed':
        occupation = random.choice([o for o in OCCUPATIONS if o not in ['Gig/Freelancers']])
        contract_type = random.choice([ct for ct in CONTRACT_TYPES if ct not in ['Zero-hour']])
        company_name = random.choice([c for c in COMPANIES if c not in ['Freelance', 'Small Business']])
        number_of_jobs = random.randint(1, 2)
        income_range = INCOME_RANGES.get(occupation, (20000, 70000))
        basic_income = random.randint(income_range[0], income_range[1])
        additional_income = random.randint(0, basic_income // 4) # Additional income is usually less than basic
    elif employment_status == 'Self-employed':
        occupation = 'Gig/Freelancers' if random.random() < 0.5 else random.choice([o for o in OCCUPATIONS if o not in ['NHS/Healthcare', 'Education/Public']])
        contract_type = random.choice(['Contract', 'Fixed-term'])
        company_name = random.choice(['Freelance', 'Small Business'] + [c for c in COMPANIES if c not in ['NHS Trust', 'Local Council']])
        number_of_jobs = 1
        income_range = INCOME_RANGES.get(occupation, (15000, 100000))
        basic_income = random.randint(income_range[0], income_range[1])
        additional_income = random.randint(0, basic_income // 2) # Additional income can be higher for self-employed
    elif employment_status == 'Retired':
        basic_income = random.randint(10000, 30000) # Pension income
        additional_income = random.randint(0, 10000) # Savings/Investments
    elif employment_status == 'Student':
        basic_income = random.randint(0, 10000) # Grants/Loans
        additional_income = random.randint(0, 15000) # Part-time job
    elif employment_status == 'Unemployed':
        basic_income = random.randint(0, 5000) # Benefits
        additional_income = random.randint(0, 5000)

    total_income = basic_income + additional_income
    property_value = round(random.uniform(150_000, 3_000_000), 2) # More realistic UK range
    deposit_percentage_value = random.uniform(0.05, 0.9)
    deposit = round(deposit_percentage_value * property_value, 2)
    deposit_percentage = round(deposit_percentage_value * 100, 2)
    loan_to_value = round(100 - deposit_percentage, 2)
    total_borrowing_amount = round(property_value - deposit, 2)

    # More realistic income to loan ratio calculation and filtering
    income_to_loan_ratio = round(total_income / (total_borrowing_amount if total_borrowing_amount > 0 else 1), 2) # Avoid division by zero

    credit_score = random.randint(300, 850) # Standard range

    start_date = fake.date_between(start_date='-30y', end_date='today').strftime("%b-%y")
    base_rate = get_base_rate(start_date)

    # More nuanced approval logic based on multiple factors
    approved = 'No'
    approval_reason = "Did not meet criteria"

    # Basic approval checks (can be expanded)
    if total_income > 15000 and loan_to_value < 95 and credit_score > 550:
        if employment_status in ['Employed', 'Self-employed']:
            if income_to_loan_ratio > 0.1: # Simple affordability check
                 approved = 'Yes'
                 approval_reason = "Meets basic income, LTV, and credit criteria"
            else:
                 approval_reason = "Low income to loan ratio"
        elif employment_status == 'Retired' and total_income > 20000 and loan_to_value < 80:
             approved = 'Yes'
             approval_reason = "Meets retired income, LTV, and credit criteria"
        elif employment_status == 'Student' and additional_income > 10000 and loan_to_value < 70:
             approved = 'Yes'
             approval_reason = "Meets student income, LTV, and credit criteria"
        elif employment_status == 'Unemployed' and additional_income > 5000 and loan_to_value < 60 and credit_score > 700:
             approved = 'Yes'
             approval_reason = "Meets unemployed income, LTV, and high credit criteria"
        else:
            approval_reason = "Did not meet specific employment status criteria"
    else:
        approval_reason = "Failed basic income, LTV, or credit score checks"


    area = random.choice(AREAS)
    repayment_type = random.choice(REPAYMENT_TYPES)
    repayment_risk_score = random.choice(REPAYMENT_RISK_SCORES)
    job_risk_score_float = round(random.uniform(0.0, 1.0), 2) # Keep for potential future use
    geo_spatial_risk_score = round(random.uniform(0.0, 1.0), 2) # Keep for potential future use

    interest_rate = None
    interest_rate_band = "N/A"

    area_risk = AREA_RISK_MAP.get(area, 'Medium') # Default to Medium if area not in map
    job_risk = JOB_RISK_MAP.get(occupation, 'High') if occupation != 'None' else 'High'
    company_risk = COMPANY_RISK_MAP.get(company_name, 'High') if company_name != 'None' else 'High'


    if approved == 'Yes':


        # Combine risk factors - simple additive model for demonstration
        total_risk_adjustment_value = (
            risk_adjustment.get(area_risk, 0) +
            risk_adjustment.get(job_risk, 0) +
            risk_adjustment.get(company_risk, 0) +
            (risk_adjustment.get(repayment_risk_score, 0) * 0.5) # Repayment risk has less weight
        )

        # Adjust for credit score - higher credit score means lower rate adjustment
        credit_score_adjustment = ((850 - credit_score) / 550) * 0.5 # Max 0.5 adjustment for lowest score
        total_risk_adjustment_value -= credit_score_adjustment

        interest_rate = round(base_rate + total_risk_adjustment_value, 2)

        interest_rate_band = (
            "Low" if interest_rate < base_rate else
            "Medium" if interest_rate < base_rate + 1.5 else # Example bands
            "High"
        )

        # Ensure interest rate is not negative or unrealistically low
        interest_rate = max(interest_rate, base_rate * 0.8) # Example lower bound

    # Generate new data points
    passport_details = {
        "passport_number": fake.passport_number(),
        "issue_date": fake.date_object().strftime("%Y-%m-%d"),
        "expiry_date": fake.date_object(random.randint(1, 365 * 10)).strftime("%Y-%m-%d"), # Corrected date_object arguments
        "country_of_issue": random.choice(NATIONALITIES) # Can be different from nationality
    } if random.random() < 0.8 else {} # Simulate some missing data

    social_security_number = generate_ni_number() if random.random() < 0.9 else None # Simulate some missing data

    transunion_data = simulate_transunion_data(credit_score) if random.random() < 0.95 else {} # Simulate some missing data


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
        "basic_yearly_income": f"£{basic_income:,.0f}", # Formatted as integer
        "additional_income": f"£{additional_income:,.0f}", # Formatted as integer
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
        "who_is_applying": random.choice(["Just me", "Joint applicant"]), # Added joint applicant
        "mortgage_approved": approved,
        "property_area": area,
        "area_risk_score": area_risk, # Use the assigned risk level
        "job_risk_score": job_risk,   # Use the assigned risk level
        "company_risk_score": company_risk, # Use the assigned risk level
        "base_rate": f"{base_rate}%",
        "interest_rate": f"{interest_rate}%" if interest_rate is not None else "N/A",
        "repayment_type": repayment_type,
        "repayment_risk_score": repayment_risk_score,
        "job_risk_score_float": job_risk_score_float, # Keep for potential future use
        "geo_spatial_risk_score": geo_spatial_risk_score, # Keep for potential future use
        "income_to_loan_ratio": income_to_loan_ratio,
        "approval_reason": approval_reason,
        "interest_rate_band": interest_rate_band,
        "passport_details": json.dumps(passport_details), # Store as JSON string
        "social_security_number": social_security_number,
        "transunion_data": json.dumps(transunion_data) # Store as JSON string
    }

# Generate dataset
data = [generate_row() for _ in range(NUM_ROWS)]
df = pd.DataFrame(data)