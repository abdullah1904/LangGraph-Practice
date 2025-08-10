from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b"
)

class LLMState(TypedDict):
    question: str
    answer: str

def llm_qa(state:LLMState)->LLMState:
    response = model.invoke(state['question'])
    state['answer'] = response.content
    return state

graph_builder = StateGraph(LLMState)

# Nodes of Graph
graph_builder.add_node("llm_qa", llm_qa)

# Edges of Graph
graph_builder.add_edge(START, "llm_qa")
graph_builder.add_edge("llm_qa", END)

graph = graph_builder.compile()

print(graph.get_graph().draw_ascii())

response = graph.invoke({"question": "What is the capital of France?"})
print(response)