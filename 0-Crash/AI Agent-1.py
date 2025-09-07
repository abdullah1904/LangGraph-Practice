from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START,END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

modal = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

def process(state:AgentState)->AgentState:
    response = modal.invoke(state['messages'])
    print("\nAI: ", response.content)
    return state

graph = StateGraph(AgentState)

graph.add_node("process",process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 

app = graph.compile()

query = input("Enter your message: ")
while query.lower() != "exit":
    app.invoke({
        "messages": [HumanMessage(content=query)]
    })
    query = input("Enter your message: ")