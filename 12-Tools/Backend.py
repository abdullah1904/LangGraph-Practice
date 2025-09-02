from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages 
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests

load_dotenv()

model = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    max_tokens=8096,
    temperature=0.3,
    reasoning_format='parsed'
)

search_tool = DuckDuckGoSearchRun(region="us")

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') using Alpha Vantage with API key in the URL.
    Args:
        symbol (str): 
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

tools = [search_tool, get_stock_price]

model_with_tools = model.bind_tools(tools)

class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chatNode(state: ChatbotState) -> ChatbotState:
    response = model_with_tools.invoke(state['messages'])
    return {
        "messages": [response]
    }

tool_node = ToolNode(tools)

graph_builder = StateGraph(ChatbotState)

graph_builder.add_node("chat_node", chatNode)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chat_node")
graph_builder.add_conditional_edges("chat_node", tools_condition)
graph_builder.add_edge("tools", "chat_node")

conn = sqlite3.connect('chatbot.db', check_same_thread=False)

checkPointer = SqliteSaver(conn=conn)

graph = graph_builder.compile(checkpointer=checkPointer)

# for message_chunk, meta in graph.stream(
#     { "messages": [HumanMessage(content="What is my name?")]},
#     config={'configurable': {'thread_id': "thread_1"}},
#     stream_mode='messages'
# ):
#     if message_chunk.content:
#         print(message_chunk.content, end='', flush=True)

def listThreads():
    allthreads = set()
    for checkPoint in checkPointer.list(None):
        allthreads.add(checkPoint.config['configurable']['thread_id'])

    return list(allthreads)
