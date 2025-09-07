from typing import TypedDict, List
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    name: str
    age: str
    skills: List[str]
    final: str

def first_node(state: AgentState) -> AgentState:
    '''
    This is the first node of the graph sequence.
    '''
    state['final'] = f'Hi {state["name"]}!' 
    return state

def second_node(state: AgentState) -> AgentState:
    '''
    This is the second node of the graph sequence.
    '''
    state['final'] = state['final'] + f' You are {state['age']} years old!' 
    return state

def third_node(state: AgentState) -> AgentState:
    '''
    This is the third node of the graph sequence.
    '''
    state['final'] = state['final'] + f' You have skills: {", ".join(state["skills"])}!'
    return state

graph = StateGraph(AgentState) 

graph.add_node("first_node", first_node)
graph.add_node("second_node", second_node)
graph.add_node("third_node", third_node)

graph.set_entry_point("first_node")
graph.add_edge("first_node", "second_node")
graph.add_edge("second_node", "third_node")
graph.set_finish_point("third_node")

app = graph.compile()

result = app.invoke({
    'age': '25',
    'name': 'Muhammad Abdullah',
    'skills': ['Python', 'JavaScript', 'Machine Learning']
})
print(result['final'])