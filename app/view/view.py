from fastapi import APIRouter
from mortgage.app.model.chat_model import ChatRequest, ChatResponse
from mortgage.app.controller.chat_controller import chatcontroller
from mortgage.src.utils.agents.agent import steer_mortgage_agent
from mortgage.app.controller.chat_controller import steer_rate_of_interest_controller
from mortgage.src.utils.helper.ocr import text_extract

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    response = chatcontroller(request.user_input)
    return ChatResponse(response=response)

@router.get("/aip", response_model=ChatResponse)
def aip_endpoint(request: ChatRequest):
    """Endpoint to Get AIP result."""
    response = steer_rate_of_interest_controller(request.user_input)
    return ChatResponse(response=str(response))

@router.post("/ocr", response_model=ChatResponse)
def ocr_endpoint(request: ChatRequest):
    """Endpoint to handle OCR requests."""
    # Placeholder for OCR processing logic
    response = text_extract(request.user_input, request.doc_type)
    return ChatResponse(response=response)

