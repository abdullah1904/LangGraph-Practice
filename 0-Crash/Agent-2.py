from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph
import math

class AgentState(TypedDict):
    values: List[int]
    name: str
    result: str
    operation: Literal['+', '*']

def process_values(state: AgentState) -> AgentState:
    '''
    This function handles multiple values in the state,
    processes them, and updates the state with the result.
    '''
    if state['operation'] == '+':
        state['result'] = f'Hi {state["name"]}, Your sum = {sum(state["values"])}.'
    elif state['operation'] == '*':
        state['result'] = f'Hi {state["name"]}, Your product = {math.prod(state["values"])}.'
    return state

graph = StateGraph(AgentState)

graph.add_node("process_values", process_values)

graph.set_entry_point("process_values")
graph.set_finish_point("process_values")

app = graph.compile()

result = app.invoke({
    "values": [1, 2, 3, 4, 5],
    "name": "Abdullah",
    "operation": "+",   
})
print(result['result'])