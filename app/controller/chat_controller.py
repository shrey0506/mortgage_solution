from mortgage.src.utils.conversation.chat import chat

def chatController(user_input: str) -> str:
    """Controller function to handle chat requests."""
    return chat(user_input = user_input)