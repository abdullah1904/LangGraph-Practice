from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    message: Optional[str] | None
    name: str

def greeting_node(state: AgentState) -> AgentState:
    '''
    Simple node that sets a greeting message in the state.
    '''
    state["message"] = "Hey" + state["name"] + "! how are you?"
    return state

graph = StateGraph(AgentState)

graph.add_node("greeting", greeting_node)

graph.set_entry_point("greeting")
graph.set_finish_point("greeting")

app = graph.compile()

result = app.invoke({'name': ' Abdullah'})
print(result)