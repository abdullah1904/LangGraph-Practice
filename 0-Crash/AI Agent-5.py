from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END, START
from typing import  TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from operator import add as add_messages
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import  PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import  Chroma
from langchain_core.tools import tool

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

embedding=HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-large-en-v1.5",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)

pdf_path = "Report.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at {pdf_path}")

loader = PyPDFLoader(pdf_path)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(documents)

db_path = "./chroma_db"
collection_name = "report"

if not os.path.exists(db_path):
    os.makedirs(db_path)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=db_path,
    collection_name=collection_name
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

@tool
def retriever_tool(query: str) -> str:
    """
        This tool searches and returns the information from the document based on the query.
        Args:
            query (str): The query to search in the document.
        Returns:
            str: The content of the document that matches the query.
    """
    results = retriever.invoke(query)
    if not results:
        return "No relevant information found in the document."
    return "\n".join([doc.page_content + "\n\n" for doc in results])

tools = [retriever_tool]

model_tools = model.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

SYSTEM_PROMPT = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}

def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    message = model_tools.invoke(messages)
    return {'messages': [message]}

def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_edge(START, "llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")

app = graph.compile()

def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = app.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()