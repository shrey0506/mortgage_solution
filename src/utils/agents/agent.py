from mortgage.sc.model import GeminiModelWrapper
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from typing import List, Optional, Union
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from mortgage.src.utils.agents.agent import get_macro_economics, reject_mortgage

global model
model = GeminiModelWrapper()

def steer_mortgage_agent():
    macro_agent = create_react_agent(
        model,
        tools=[get_macro_economics, reject_mortgage],
        prompt=(
            "You are a financial decision agent. First, call get_macro_economics to retrieve data. "
            "Then, pass the result to \"reject_mortgage\" to make a decision. Do not skip steps or call them in parallel."
        ),
        name="MortgageAgent"
    )
    # Define the workflow
    workflow = create_swarm([macro_agent], default_active_agent="MortgageAgent")
    app = workflow.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    response = app.invoke({"messages": [{"role": "user", "content": "should I approve the mortgage?"}]}, config)
    print(response)