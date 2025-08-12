from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct", 
    max_tokens=1024,
    temperature=0.5,  
)

# Structured Output Schema and Models
class SentimentSchema(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Sentiment of the review")


class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description='The category of issue mentioned in the review')
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')
    urgency: Literal["low", "medium", "high"] = Field(description='How urgent or critical the issue appears to be')

structured_model_sentiment = model.with_structured_output(SentimentSchema)
structured_model_diagnosis = model.with_structured_output(DiagnosisSchema)

# Prompt Templates
findSentimentPrompt = PromptTemplate(
    input_variables=["review"],
    template="Analyze the sentiment of the following review and classify it as positive or negative: \n {review}"
)

positiveReplyPrompt = PromptTemplate(
    input_variables=["review"],
    template="You are a support assistant. Write a warm thank-you message in response to this review:\n\n\"{review}\"\nAlso, kindly ask the user to leave feedback on our website."
)

diagnosisPrompt = PromptTemplate(
    input_variables=["review"],
    template="""Diagnose this negative review:\n\n{review}\n""Return issue_type, tone, and urgency."""
)

negativeReplyPrompt = PromptTemplate(
    input_variables=["issue_type", "tone", "urgency"],
    template="You are a support assistant. The user had a '{issue_type}' issue, sounded '{tone}', and marked urgency as '{urgency}'.Write an empathetic, helpful resolution message."
)

# LLM Chains
findSentimentChain = findSentimentPrompt | structured_model_sentiment 
positiveReplyChain = positiveReplyPrompt | model | StrOutputParser()
diagnosisChain = diagnosisPrompt | structured_model_diagnosis
negativeReplyChain = negativeReplyPrompt | model | StrOutputParser()

# Graph State
class ReviewHandlerState(TypedDict):
    review: str
    sentiment: Literal['positive', 'negative']
    diagnosis: dict
    response_reply: str

# Graph Node Operations and Condition Checkers
def findSentiment(state:ReviewHandlerState)->ReviewHandlerState:
    findSentimentResult = findSentimentChain.invoke({"review": state["review"]})
    return {
        "sentiment": findSentimentResult.sentiment
    }

def checkSentiment(state:ReviewHandlerState)->Literal['positive_reply', 'run_diagnosis']:
    if state["sentiment"] == "positive":
        return "positive_reply"
    else:
        return "run_diagnosis"

def positiveReply(state:ReviewHandlerState)->ReviewHandlerState:
    positiveReplyResult = positiveReplyChain.invoke({"review": state['review']})
    return {
        "response_reply": positiveReplyResult
    }

def runDiagnosis(state:ReviewHandlerState)->ReviewHandlerState:
    diagnosisResult = diagnosisChain.invoke({"review": state["review"]})
    return {
        "diagnosis": diagnosisResult.model_dump()
    }

def negativeReply(state: ReviewHandlerState)->ReviewHandlerState:
    negativeReplyResult = negativeReplyChain.invoke({
        "issue_type": state["diagnosis"]["issue_type"],
        "tone": state["diagnosis"]["tone"],
        "urgency": state["diagnosis"]["urgency"]
    })
    return {
        "response_reply": negativeReplyResult
    }

# Graph Building
graph_builder = StateGraph(ReviewHandlerState)

# Nodes of graph
graph_builder.add_node("find_sentiment", findSentiment)
graph_builder.add_node("positive_reply", positiveReply)
graph_builder.add_node("run_diagnosis", runDiagnosis)
graph_builder.add_node("negative_reply", negativeReply)

# Edges of graph
graph_builder.add_edge(START, "find_sentiment")
graph_builder.add_conditional_edges("find_sentiment", checkSentiment)
graph_builder.add_edge("run_diagnosis", "negative_reply")

graph_builder.add_edge("positive_reply", END)
graph_builder.add_edge("negative_reply", END)

# Graph Compiling
graph = graph_builder.compile()

# Graph View
print(graph.get_graph().print_ascii())

# Graph Invoking
response = graph.invoke({
    "review": "The product was so bad that I will never buy it again. It was a waste of money and time."
    # "review": "I love the new features in the app! It's so user-friendly and has made my life easier. Thank you for the great work!"
})

print("Sentiment:", response["sentiment"])
print("Response Reply:", response["response_reply"])