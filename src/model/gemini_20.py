from typing import List, Optional, Any
import google.generativeai as genai
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import Tool
from pydantic import PrivateAttr


class GeminiModelWrapper(BaseChatModel):
    sys_instruct: str = "You are a helpful assistant."

    # ✅ Declare internal-only (non-validated) attributes
    _tools: Optional[List[Tool]] = PrivateAttr(default=None)
    _model: Any = PrivateAttr(default=None)

    def __init__(self, sys_instruct: str = "You are a helpful assistant."):
        super().__init__()
        self.sys_instruct = sys_instruct

        # ✅ Configure API key for Google Gemini
        genai.configure(api_key="AIzaSyB7zB8ZzqXoLZ-hkH2iMtgiHsyTemAB6-0")

        # ✅ Store model reference in private attribute
        self._model = genai.GenerativeModel("gemini-1.5-flash")  # or "gemini-pro"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        chat_history = [{"role": "model", "parts": [self.sys_instruct]}]
        for msg in messages:
            role = "user" if msg.type == "human" else "model"
            chat_history.append({"role": role, "parts": [msg.content]})

        # ✅ Call the Gemini API
        response = self._model.generate_content(chat_history)

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=response.text))]
        )

    def bind_tools(self, tools: List[Tool], **kwargs: Any) -> "GeminiModelWrapper":
        self._tools = tools
        return self

    @property
    def _llm_type(self) -> str:
        return "gemini-api"
