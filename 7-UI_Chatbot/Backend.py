from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages 
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    max_tokens=1024,
    temperature=0.7
)

class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chatNode(state: ChatbotState) -> ChatbotState:
    response = model.invoke(state['messages'])
    return {
        "messages": [response]
    }


graph_builder = StateGraph(ChatbotState)

graph_builder.add_node("chat_node", chatNode)

graph_builder.add_edge(START, "chat_node")
graph_builder.add_edge("chat_node", END)

checkPointer = InMemorySaver()

graph = graph_builder.compile(checkpointer=checkPointer)