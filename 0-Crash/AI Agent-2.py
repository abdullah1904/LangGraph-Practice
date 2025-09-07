from typing import List, TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

modal = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

def process(state:AgentState)->AgentState:
    '''This node will process the state and return it.'''
    response = modal.invoke(state['messages'])
    state['messages'].append(AIMessage(content=response.content))
    print('\nAI: ', response.content)
    return state

graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()

conversation_history = []

query = input("You: ")
while query.lower() != "exit":
    conversation_history.append(HumanMessage(content=query))
    result = app.invoke({
        "messages": conversation_history
    })
    conversation_history = result['messages']
    query = input("You: ")

with open("conversation_history.txt", "w") as f:
    f.write("Conversation History:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n\n")
    f.write("End of conversation.\n")