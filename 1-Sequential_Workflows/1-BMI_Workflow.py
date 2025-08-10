from langgraph.graph import StateGraph, START,END
from typing import TypedDict

# Graph State
class BMIState(TypedDict):
    weight_kg: float
    height_m: float 
    bmi: float
    category: str

# BMI Calculation Node
def calculate_bmi(state: BMIState) -> BMIState:
    weight = state["weight_kg"]
    height = state["height_m"]
    state['bmi'] = weight / (height ** 2)
    return state

def label_bmi(state:BMIState)->BMIState:
    bmi = state['bmi']
    if bmi < 18.5:
        state['category'] = 'Underweight'
    elif 18.5 <= bmi < 24.9:
        state['category'] = 'Normal weight'
    elif 25 <= bmi < 29.9:
        state['category'] = 'Overweight'
    else:
        state['category'] = 'Obesity'
    return state

graph_builder = StateGraph(BMIState)

# Nodes of Graph
graph_builder.add_node("calculate_bmi", calculate_bmi)
graph_builder.add_node("label_bmi", label_bmi)

# Edges of Graph
graph_builder.add_edge(START, "calculate_bmi")
graph_builder.add_edge("calculate_bmi", "label_bmi")
graph_builder.add_edge("label_bmi", END)

graph = graph_builder.compile()

print(graph.get_graph().draw_ascii())

response = graph.invoke({"height_m": 1.73, "weight_kg": 80})
print(response)