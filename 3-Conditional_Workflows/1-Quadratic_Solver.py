from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal

class QuadraticSolver(TypedDict):
    a: int
    b: int
    c: int
    equation: str
    discriminant: float
    result: str

graph_builder = StateGraph(QuadraticSolver)


def showEquation(state: QuadraticSolver)->QuadraticSolver:
    equation = f"{state['a']}x^2 + {state['b']}x + {state['c']} = 0"
    return {
        'equation': equation
    }

def calculateDiscriminant(state: QuadraticSolver) -> QuadraticSolver:
    discriminant = state['b']**2 - 4*state['a']*state['c']
    return {
        'discriminant': discriminant
    }

def realRoots(state:QuadraticSolver)->QuadraticSolver:
    root1 = (-state['b'] + state['discriminant']**0.5) / (2*state['a'])
    root2 = (-state['b'] - state['discriminant']**0.5) / (2*state['a'])
    result = f"Roots are {root1} and {root2}"
    return {
        'result': result
    }

def repeatedRoots(state:QuadraticSolver)->QuadraticSolver:
    root = -state['b'] / (2 * state['a'])
    result = f"Only Root is {root}"
    return {
        'result': result
    }

def noRealRoots(state:QuadraticSolver)->QuadraticSolver:
    result = "No real roots"
    return {
        'result': result
    }

def discriminantTypeChecker(state: QuadraticSolver) -> Literal['real_roots', 'repeated_roots', 'no_real_roots']:
    if state['discriminant'] > 0:
        return 'real_roots'
    elif state['discriminant'] == 0:
        return 'repeated_roots'
    else:
        return 'no_real_roots'

# Nodes of graph
graph_builder.add_node("show_equation", showEquation)
graph_builder.add_node("calculate_discriminant", calculateDiscriminant)
graph_builder.add_node("real_roots", realRoots)
graph_builder.add_node("repeated_roots", repeatedRoots)
graph_builder.add_node("no_real_roots", noRealRoots)

# Edges of graph
graph_builder.add_edge(START, "show_equation")
graph_builder.add_edge("show_equation", "calculate_discriminant")
graph_builder.add_conditional_edges("calculate_discriminant", discriminantTypeChecker)

graph_builder.add_edge("real_roots", END)
graph_builder.add_edge("repeated_roots", END)
graph_builder.add_edge("no_real_roots", END)

graph = graph_builder.compile()

print(graph.get_graph().print_ascii())

response = graph.invoke({
    'a': 4,
    'b': -5,
    'c': -4
})

print(response)