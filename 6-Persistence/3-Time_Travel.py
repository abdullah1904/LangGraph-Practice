from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict
from langchain.prompts import PromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    max_tokens=120,
    temperature=0.7
)

generatePrompt = PromptTemplate(
    template="Generate a joke on the topic {topic}",
    input_variables=['topic']
)

explainPrompt = PromptTemplate(
    template="Write an explanation for the joke - {joke}",
    input_variables=['joke']
)

generateChain = generatePrompt | model | StrOutputParser()
explainChain = explainPrompt | model | StrOutputParser()

class JokeGeneratorState(TypedDict):
    topic: str
    joke: str
    explanation: str

def generateJoke(state: JokeGeneratorState)->JokeGeneratorState:
    joke = generateChain.invoke({"topic": state["topic"]})
    return {
        "joke": joke
    }

def explainJoke(state: JokeGeneratorState)->JokeGeneratorState:
    explanation = explainChain.invoke({"joke": state["joke"]})
    return {
        "explanation": explanation
    }

graph_builder = StateGraph(JokeGeneratorState)

graph_builder.add_node("generate_joke", generateJoke)
graph_builder.add_node("explain_joke", explainJoke)

graph_builder.add_edge(START, "generate_joke")
graph_builder.add_edge("generate_joke", "explain_joke")
graph_builder.add_edge("explain_joke", END)

checkPointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkPointer)

config = {
    "configurable": {
        "thread_id": "1"
    }
}

response = graph.invoke({'topic':'pizza'}, config=config)


print("-- Response --")
print(graph.get_state(config))
print(list(graph.get_state_history(config)))

print(graph.get_state({
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": list(graph.get_state_history(config))[2].config['configurable']['checkpoint_id']
    }
}))

print(graph.invoke(None,{
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": list(graph.get_state_history(config))[2].config['configurable']['checkpoint_id']
    }
}))

print(len(list(graph.get_state_history(config))))