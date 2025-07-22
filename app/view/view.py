from fastapi import APIRouter
from mortgage.app.model.chat_model import ChatRequest, ChatResponse
from mortgage.app.controller.chat_controller import chatcontroller

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    response = chatcontroller(request.user_input)
    return ChatResponse(response=response)