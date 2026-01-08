from pyexpat.errors import messages
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') using Alpha Vantage with API key in the URL.
    Args:
        symbol (str): Stock symbol to fetch the price for.
    Returns:
        dict: JSON response containing stock price information.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Purchase a given quantity of stock for a specified symbol.
    Args:
        symbol (str): Stock symbol to purchase.
        quantity (int): Number of shares to purchase.
    Returns:
        dict: Confirmation of purchase or cancellation.
    """
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }


tools = [get_stock_price, purchase_stock]
model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.7,
    max_tokens=512,
    reasoning_format='parsed'
).bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    config = {
        "configurable": {
            'thread_id': "stock-agent-001"
        }
    }

    response = chatbot.invoke({
        "messages": [HumanMessage(content=user_input)],
    }, config=config)

    interrupts = response.get("__interrupt__", [])
    if interrupts:
        print(interrupts[0].value)
        decision = input("Your decision: ").strip().lower()
        response = chatbot.invoke(Command(resume=decision), config=config)

    last_message = response["messages"][-1].content
    print(f"Bot: {last_message}\n\n")