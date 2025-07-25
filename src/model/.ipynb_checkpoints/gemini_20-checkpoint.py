from google import genai
from google.genai import types
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    HttpOptions,
    Tool as GeminiTool,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import Tool
from typing import List, Optional, Any

class GeminiModelWrapper(BaseChatModel):
    sys_instruct: str = "You are a helpful assistant."
    _tools: Optional[List[Tool]] = None

    def __init__(self, sys_instruct: str = "You are a helpful assistant."):
        super().__init__()
        setattr(self, "sys_instruct", sys_instruct)
        setattr(self, "_tools", None)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = genai.Client(vertexai=True, project="ltc-reboot25-team-56", location="global")
        model = "gemini-2.5-flash"
        contents = [
            types.Content(role="model", parts=[types.Part.from_text(text=self.sys_instruct)]),
        ]
        for msg in messages:
            role = "user" if msg.type == "human" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))

        generate_content_config = types.GenerateContentConfig(
            temperature=1, top_p=1, seed=0, max_output_tokens=2048,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ],
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )

        response = client.models.generate_content(
            model=model, contents=contents,
            config=generate_content_config,
        )

        if hasattr(response, "candidates"):
            candidate = response.candidates[0]
            if hasattr(candidate, "function_call"):
                tool_call = candidate.function_call
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(content="", additional_kwargs={"tool_calls": [{"name": tool_call.name, "args": tool_call.args,}]},)
                        )
                    ]
                )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=response.text))]
        )

    def bind_tools(self, tools: List[Tool], **kwargs: Any) -> "GeminiModelWrapper":
        object.__setattr__(self, "_tools", tools)
        return self

    @property
    def _llm_type(self) -> str:
        return "vertex-gemini"