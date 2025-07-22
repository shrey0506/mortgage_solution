From mortgage.sc.model gemini_20 import GeminiModelwrapper From langgraph. checkpoint.memory import InMemorySaver from langgraph. prebuilt import create
react
_agent
from langgraph
_swarm import create _handoff_tool, create_ swarm
From langchain google_genai import ChatGoogleGenerativeAI from langgraph. graph import StateGraph from langchain core.messages import HumanMessage global model
model= GeminiModelwrapper ()
from typing import List, Optional, Union from langchain_core.messages import BaseMessage from pydantic import BaseModel
model = GeminiModelWrapper
def add(a: int, b: int) →› int:
'Add two numbers'"
return a + b
def test_agent () :
alice = create_react_agent (
model,
tools=[add, create _handoff_tool (agent_ name= "Bob")],
prompt="You are Alice, an addition expert. If the user wants to speak to Bob, use the handoff tool.", name="Alice"
bob = create_react_agent
model, tools=[create_handoff_tool (agent_name="Alice"
', description="Transfer to Alice for math help.")],
prompt="You are Bob, you speak like a pirate. name="Bob"
5
workflow = create swarm([alice, bob], default_active_agent="Alice")
app = workflow.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "1"}}
response = app.invoke({"messages": [{"role": "user", "content": "I'd like to speak to Bob"}l}, config)
print (response)