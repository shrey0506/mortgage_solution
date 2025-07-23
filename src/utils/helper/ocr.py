from google import genai
from google.generativeai import types
from google.generativeai.types import (
    GenerateContentConfig, GoogleSearch, HttpOptions, Tool, Blob
)
from google.api_core.client_options import ClientOptions
import mimetypes
import json
import re
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from mortgage.src.model.gemini_image import image_to_text_with_gemini
from mortgage.src.utils.helper.boe_interest import fetch_latest_interest_rate

def text_extract(image_path: str, doc_type: str) -> str:
    """
    Extracts structured text data from a scanned document image related to mortgage processing.

    Args:
        image_path: The local path to the image.
        doc_type: Type of document being processed (e.g. passport, bank_ statement, payslip).

    Returns:
        str: A JSON-formatted string with extracted fields.
    """
    try:
        # Load image data
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()

        # Guess MIME type
        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type = mime_type or "image/png"

        # System-level instruction
        system_instruction = f"""
        **You are an intelligent assistant for processing UK mortgage application documents.**
        Your task is to extract structured key information from a scanned image of a '{doc_type}'.

        Instructions:
        - Use OCR to read and interpret the image.
        - Depending on the document type, extract and label relevant data fields using lowercase JSON keys with underscores.
        - Format all dates as YYYY-MM-DD.
        - If any fields are illegible or not present, mark them as null.
        - Return only a valid JSON object without extra commentary.

        Document-specific fields to extract:
        1. Passport
            - full_name
            - date_of_birth
            - place_of_birth
            - nationality
            - gender
            - passport_number
            - issue_date
            - expiry_date
        2. Driving Licence
            - full_name
            - date_of_address
            - licence_number
            - issue_date
            - expiry_date
            - endorsements (if visible)
        3. Utility Bill
            - full_name
            - address
            - billing_date
            - provider_name
            - account_number
            - bill_amount
        4. Salary Slip
            - full_name
            - pay_period_start
            - pay_period_end
            - net_pay
            - gross_pay
            - national_insurance_number
            - employer_name
        5. Bank Statement
            - full_name
            - account_holder_name
            - account_number
            - sort_code
            - statement_start_date
            - statement_end_date
            - transaction_summary (structured list of date, description, and amount)
        6. P60 Tax Form
            - full_name
            - national_insurance_number
            - tax_year_end
            - employer_name
            - total_pay
            - tax_paid
        7. Property Data
            - owner_name
            - property_address
            - property_type
            - valuation_amount
            - purchase_date
            - property_id_reference

        Ensure accuracy in field labeling. Prioritize clearly readable content. Output should strictly follow JSON formatting conventions as described.
        """

        # Pass to Gemini model
        model = genai.GenerativeModel('gemini-pro-vision', system_instruction=system_instruction)
        result = model.generate_content(
            contents=[HumanMessage(content="Please analyze this document for mortgage evaluation."), {"mime_type": mime_type, "data": image_data}]
        )
        return result.text

    except Exception as e:
        return f"Error during OCR extraction: {str(e)}"