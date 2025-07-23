from langchain.tools import tool
from langchain_core.messages import HumanMessage
from mortgage.src.model.gemini_20 import GeminiModelWrapper
from mortgage.src.utils.helper.boe_interest import fetch_latest_interest_rate

global model
model = GeminiModelWrapper()


def acct_reject_based_on_credit_history(credit_history):
    """
    Evaluates a given credit history against predefined rejection criteria for loan or mortgage applications.

    Parameters:
        credit_history (dict): A dictionary containing the applicant's credit and financial information.

    Returns:
        str: A decision message indicating whether the application is accepted or rejected based on the criteria.
    """
    # The function uses a set of predefined rejection criteria across
    # several categories:
    # - Identity & Residency
    # - Income & Employment
    # - Deposit & Property Value
    # - Credit History
    # - Affordability
    # - Application Consistency
    # Each category includes specific conditions that may lead to rejection. Some conditions may require additional documentation or stronger financial standing to override.
    criteria_for_rejection = {
        "Identity & Residency": [
            "Not registered to vote",
            "No UK bank account",
            "Insufficient UK residency duration (less than 2 years)",
            {
                "Condition": "If residency is less than 2 years",
                "Additional Requirements": [
                    "Provide a larger deposit (e.g. more than 10%)",
                    "Show strong employment stability",
                    "Demonstrate good UK credit history"
                ]
            }
        ],
        "Income & Employment": [
            "Income below minimum threshold",
            "Unstable employment (e.g., short-term contracts)",
            "Self-employed without sufficient documentation"
        ],
        "Deposit & Property Value": [
            "Deposit below required percentage (typically <5-10%)",
            "Property value mismatch with income/deposit"
        ],
        "Credit History": [
            "Poor credit score",
            "Missed payments or defaults",
            "County Court Judgements (CCJs)",
            "Use of payday loans"
        ],
        "Affordability": [
            "High debt-to-income ratio",
            "Monthly outgoings exceed affordability limits"
        ],
        "Application Consistency": [
            "Discrepancies between initial and full application",
            "Unverifiable or inconsistent information"
        ]
    }
    # This function currently only defines the criteria.
    # Add logic here to evaluate the credit_history against criteria_for_rejection.
    # For now, it returns a placeholder message.
    return "Evaluation logic for credit history is not yet implemented."



def aip_criteria_for_accpt_reject():
    """
    Returns simplified AIP rejection criteria across key categories.
    """
    return {
        "Identity & Residency": [
            "Not registered to vote",
            "No UK bank account",
            "UK residency < 2 years",
            {"If residency < 2 years": [
                "Deposit > 10%",
                "Stable employment",
                "Good UK credit history"
            ]}
        ],
        "Income & Employment": [
            "Low income",
            "Unstable job (e.g. short-term contracts)",
            "Self-employed without docs"
        ],
        "Deposit & Property": [
            "Deposit < 5-10%",
            "Property value mismatch"
        ],
        "Affordability": [
            "High debt-to-income ratio",
            "Outgoings exceed limits"
        ],
        "Application Consistency": [
            "Info mismatch",
            "Unverifiable details"
        ]
    }


def reject_mortgage(user_details: str = None, credit_score_check=None):
    """
    Uses AIP or credit score criteria to assess mortgage eligibility and suggest an interest rate.
    """
    # Determine which criteria to use
    criteria = aip_criteria_for_accpt_reject() if credit_score_check is None else acct_reject_based_on_credit_history(credit_score_check)

    system_prompt = f"""
    You are a senior mortgage analyst. Review the user's details and decide whether to accept or reject the application.
    Use the following criteria: {criteria}

    Provide:
    1. Initial interest rate (based on macroeconomics and risk assessment if applicable, otherwise suggest a rate based on criteria)
    2. Decision: Accept or Reject
    3. Reasoning for the decision.
    """

    user_prompt = f"""
    Based on the user's details below, provide:
    1. Initial interest rate
    2. Decision: Accept or Reject
    3. Reasoning for the decision.

    User Details: {user_details}
    """

    model.sys_instruct = system_prompt
    messages = [HumanMessage(content=user_prompt)]
    result = model.generate(messages)

    # Validate response
    if not result or not result.generations or not result.generations[0].message.content:
        return "Error: No response generated by the model."

    response = result.generations[0].message.content
    return response