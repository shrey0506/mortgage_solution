from mortgage.sc.model import GeminiModelWrapper
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from typing import List, Optional, Union
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

global model
model = GeminiModelWrapper()

def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

def test_agent():
    alice = create_react_agent(
        model,
        tools=[add, create_handoff_tool(agent_name="Bob")],
        prompt="You are Alice, an addition expert. If the user wants to speak to Bob, use the handoff tool.",
        name="Alice"
    )
    bob = create_react_agent(
        model,
        tools=[create_handoff_tool(agent_name="Alice", description="Transfer to Alice for math help.")],
        prompt="You are Bob, you speak like a pirate.",
        name="Bob"
    )

    workflow = create_swarm([alice, bob], default_active_agent="Alice")
    app = workflow.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    response = app.invoke({"messages": [HumanMessage(content="I'd like to speak to Bob")]}, config)
    print(response)
