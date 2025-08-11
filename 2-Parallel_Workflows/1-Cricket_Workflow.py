from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class CricketWorkflowState(TypedDict):
    # Input
    runs: int
    balls: int
    fours: int
    sixes: int
    # Output
    strike_rate: float
    balls_per_boundary: float
    boundary_percent: float
    summary: str


def calculateStrikeRate(state: CricketWorkflowState) -> CricketWorkflowState:
    if state['balls'] > 0:
        striker_rate = (state['runs'] / state['balls']) * 100
    else:
        striker_rate = 0
    return {"strike_rate": striker_rate}

def calculateBallsPerBoundary(state: CricketWorkflowState) -> CricketWorkflowState:
    if state["fours"] + state["sixes"] > 0:
        balls_per_boundary = state["balls"] / (state["fours"] + state["sixes"])
    else:
        balls_per_boundary = 0
    return {"balls_per_boundary": balls_per_boundary}

def calculateBoundaryPercent(state: CricketWorkflowState) -> CricketWorkflowState:
    if state["balls"] > 0:
        boundary_percent = ((state["fours"] * 4 + state["sixes"] * 6) / state["balls"]) * 100
    else:
        boundary_percent = 0
    return {"boundary_percent": boundary_percent}

def summary(state:CricketWorkflowState)->CricketWorkflowState:
    summary = f"""
    Strike Rate - {state['strike_rate']:<.2f} \n
    Balls per Boundary - {state['balls_per_boundary']:<.2f} \n
    Boundary Percent - {state['boundary_percent']:<.2f} \n
    """
    return {"summary": summary}

graph_builder = StateGraph(CricketWorkflowState)

# Nodes of Graph
graph_builder.add_node("calculate_strike_rate", calculateStrikeRate)
graph_builder.add_node("calculate_balls_per_boundary", calculateBallsPerBoundary)
graph_builder.add_node("calculate_boundary_percent", calculateBoundaryPercent)
graph_builder.add_node("summary", summary)

# Edges of Graph
graph_builder.add_edge(START, "calculate_strike_rate")
graph_builder.add_edge(START, 'calculate_balls_per_boundary')
graph_builder.add_edge(START, 'calculate_boundary_percent')

graph_builder.add_edge("calculate_strike_rate", "summary")
graph_builder.add_edge("calculate_balls_per_boundary", "summary")
graph_builder.add_edge("calculate_boundary_percent", "summary")

graph_builder.add_edge("summary", END)

graph = graph_builder.compile()

print(graph.get_graph().print_ascii())

response = graph.invoke({
    "runs": 100,
    "balls": 50,
    "fours": 6,
    "sixes": 4
})

print(response['summary'])