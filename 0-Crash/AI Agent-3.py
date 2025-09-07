from typing import Sequence, Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, ToolMessage, BaseMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add_numbers(a: int, b: int) -> int:
    """This is addition function that adds 2 numbers together."""
    return a + b

@tool
def subtract_numbers(a: int, b: int) -> int:
    """This is subtraction function that subtracts 2 numbers."""
    return a - b

@tool
def multiply_numbers(a: int, b: int) -> int:
    """This is multiplication function that multiplies 2 numbers."""
    return a * b

tools = [
    add_numbers,
    subtract_numbers,
    multiply_numbers
]

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3
).bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState): 
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState) 

graph.add_node("our_agent", model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))