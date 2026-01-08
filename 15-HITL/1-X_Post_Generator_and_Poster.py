from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.7,
    max_tokens=512,
    reasoning_format='parsed'
)

class GraphState(TypedDict):
    topic: str
    post: str
    posted: bool



def generate_node(state:GraphState) -> GraphState:
    prompt = f"Generate a social media post about the following topic: {state['topic']}"
    response = model.invoke(prompt)
    return {
        "post": response.content
    }

def post_node(state: GraphState)->GraphState:
    decision = interrupt({
        "type": "approval",
        "message": f"Do you want to post the following content?\n\n{state['post']}"
    })

    if decision == "post":
        return {
            "posted": True
        }
    else:
        return {
            "posted": False
        }

graph_builder = StateGraph(GraphState)

graph_builder.add_node("generate_node", generate_node)
graph_builder.add_node("post_node", post_node)

graph_builder.add_edge(START, "generate_node")
graph_builder.add_edge("generate_node", "post_node")
graph_builder.add_edge("post_node", END)


checkpointer = InMemorySaver()

graph = graph_builder.compile(checkpointer=checkpointer)

while True:
    input_topic = input("Enter a topic for the social media post (or 'exit' to quit): ")
    if input_topic.lower() == 'exit':
        break

    config = {
        "configurable": {
            'thread_id': "test-123"
        }
    }
    
    response = graph.invoke({
        "topic": input_topic
    },config=config)

    if '__interrupt__' in response and response['__interrupt__']:
        print(response["__interrupt__"][0].value['message'])
        approval_input = input("Type 'post' to approve or anything else to reject: ")
        final_response = graph.invoke(Command(resume=approval_input), config=config)
        if final_response.get('posted'):
            print("The post has been published!")
        else:
            print("The post was not published.")
    