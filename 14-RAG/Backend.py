from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Optional, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages 
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests

import tempfile

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()


model = ChatGroq(
    model="qwen/qwen3-32b",
    max_tokens=8096,
    temperature=0.3,
    reasoning_format='parsed'
)

embedding_model = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-large-en-v1.5",
)

vector_store = Chroma(
    persist_directory='./chroma_db',
    collection_name='my_documents',
    embedding_function=embedding_model,
)

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory='./chroma_db',
            collection_name='my_documents',
        )
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


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

@tool
def search_docs(query: str) -> str:
    """
    Search the vector database for relevant documents.
    Args:
        query (str): The search query.
    Returns:
        str: The search results.
    """
    results = vector_store.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in results])


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant documents for the given query using the thread's retriever.
    Args:
        query (str): The search query.
        thread_id (Optional[str]): The thread identifier.
    Returns:
        dict: The retrieval results including context and metadata.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

tools = [search_tool, get_stock_price, search_docs, rag_tool]

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

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})
