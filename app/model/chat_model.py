from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    response: str

class ChatModel:
    def generate_response(self, user_input: str) -> str:
        return f"{user_input}"
    
