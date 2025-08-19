from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict
import time

class CrashState(TypedDict):
    input: str
    step1: str
    step2: str
    step3: str

def step1(state: CrashState) -> CrashState:
    print("Executing step 1...")
    return {
        'step1': 'done'
    }

def step2(state: CrashState) -> CrashState:
    print("Executing step 2...")
    time.sleep(30)
    return {
        'step2': 'done'
    }

def step3(state: CrashState) -> CrashState:
    print("Executing step 3...")
    return {
        'step3': 'done'
    }

graph_builder = StateGraph(CrashState)

graph_builder.add_node("step1", step1)
graph_builder.add_node("step2", step2)
graph_builder.add_node("step3", step3)

graph_builder.add_edge(START, "step1")
graph_builder.add_edge("step1", "step2")
graph_builder.add_edge("step2", "step3")
graph_builder.add_edge("step3", END)

checkpointer = InMemorySaver()

graph = graph_builder.compile(checkpointer=checkpointer)

config={
    "configurable": {
        "thread_id": 'thread-1'
    }
}

try:
    response = graph.invoke({
        "input": "Hello"
    }, config=config)
except KeyboardInterrupt:
    print("Simulating crash...")
    print(graph.get_state(config=config))
    final_response = graph.invoke(None, config=config)
    print(graph.get_state_history(config=config))