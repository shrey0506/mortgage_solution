from mortgage.src.utils.conversation.chat import chat
from mortgage.src.utils.helper.macro_economics import get_macro_economics
from mortgage.src.utils.helper.reject_mortgage import reject_mortgage
import json
import re
from langchain.schema import HumanMessage
from mortgage.src.model.gemini_20 import GeminiModelWrapper
from mortgage.src.utils.helper.ocr import text_extract

global model
model = GeminiModelWrapper()

def chatController(user_input: str) -> str:
    return chat(user_input=user_input)

def extract_response(raw_response: str) -> str:
    """Extracts response in format of Reason, Status, and ROI"""
    print("raw response:", type(raw_response))
    raw_response = str(raw_response)
    match = re.search(r"message=AIMessage\(content='(.*?)'\)", raw_response)
    if match:
        extracted_text = match.group(1)
        return (extracted_text)
    else:
        return ("No match found.")


def steer_rate_of_interest_controller(input: str) -> str:
    """Combines agent responses and parses them into a structured plain text format.
    Args:
        input (str): User input passed to agents.
    Returns:
        str: Parsed response with Status, Reason, and Rate of Interest.
    """
    system_prompt = """
    You are a strict extraction engine.
    You are given a full mortgage application assessment.
    Your job is to return ONLY the following three lines in this exact format:
    Status: <Accepted or Rejected>
    Reason: <a short description why it was accepted or rejected>
    ROI: <Interest Rate number with % sign example 4.94%>
    Donot include any commentry, explanation, greetings, or markdown.
    Return only the three lines - Format should be
    Status: ...
    Reason: ...
    ROI: ...
    """

    accept_reject_mortgage = reject_mortgage(input)
    macro_geo_economics = get_macro_economics(input)
    clubbed_response = f"{accept_reject_mortgage} \n{macro_geo_economics}"
    user_prompt = f"From this response {clubbed_response} please give me\nstatus: ...\nreason:...\nroi: ..."

    model.sys_instruct = system_prompt
    messages = [HumanMessage(content=user_prompt)]
    result = model.generate(messages)
    result = extract_response(result)
    return result

def open_image(app_id: str, doc_type: str, doc_name: str, file_bytes: bytes) -> str:
    json_result = text_extract(file_bytes, doc_name)
    return json_result

    