from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    max_tokens=1024,
    temperature=0.7,
)

outline_prompt = PromptTemplate(
    template="Generate a detailed outline for a blog on the topic - {title}",
    input_variables=["title"]
)

content_prompt = PromptTemplate(
    template="Write a detailed blog on the title {title} using the following outline \n {outline}",
    input_variables=["title", "outline"]
)

class BlogState(TypedDict):
    title: str
    outline: str
    content: str

def generate_outline(state: BlogState)->BlogState:
    generate_outline_chain = outline_prompt | model | StrOutputParser()
    state['outline'] = generate_outline_chain.invoke({
        "title": state['title']
    })
    return state

def generate_content(state:BlogState)->BlogState:
    generate_content_chain = content_prompt | model | StrOutputParser()
    state['content'] = generate_content_chain.invoke({
        "title": state['title'],
        "outline": state['outline']
    })
    return state

graph_builder = StateGraph(BlogState)

# Nodes of Graph
graph_builder.add_node("generate_outline", generate_outline)
graph_builder.add_node("generate_content", generate_content)

# Edges of Graph
graph_builder.add_edge(START, "generate_outline")
graph_builder.add_edge("generate_outline", "generate_content")
graph_builder.add_edge("generate_content", END)

graph = graph_builder.compile()

print(graph.get_graph().draw_ascii())

response = graph.invoke({
    "title": "The Future of AI in Healthcare"
})
print(response)