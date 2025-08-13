from langgraph.graph import StateGraph, START, END
from typing import Literal, TypedDict, Annotated
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import operator

load_dotenv()

generationModel = ChatGroq(
    model="llama-3.3-70b-versatile"
)

evaluationModel = ChatGroq(
    model="moonshotai/kimi-k2-instruct"
)

optimizationModel = ChatGroq(
    model="moonshotai/kimi-k2-instruct"
)

# Structured Output Schema and Models
class EvaluationSchema(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(..., description="Final evaluation result.")
    feedback: str = Field(..., description="feedback for the tweet.")

structuredEvaluationModel = evaluationModel.with_structured_output(EvaluationSchema)

# Prompt Templates
generationPrompt = ChatPromptTemplate(
    messages=[
        ("system","You are a funny and clever Twitter/X influencer."),
        ("human","""
            Write a short, original, and hilarious tweet on the topic: "{topic}".
            Rules:
            - Do NOT use question-answer format.
            - Max 280 characters.
            - Use observational humor, irony, sarcasm, or cultural references.
            - Think in meme logic, punchlines, or relatable takes.
            - Use simple, day to day english
            """
        )
    ],
    input_variables=["topic"]
)

evaluationPrompt = ChatPromptTemplate(
    messages=[
        ("system", "You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
        ("human","""
            Evaluate the following tweet:

            Tweet: "{post}"

            Use the criteria below to evaluate the tweet:

            1. Originality – Is this fresh, or have you seen it a hundred times before?  
            2. Humor – Did it genuinely make you smile, laugh, or chuckle?  
            3. Punchiness – Is it short, sharp, and scroll-stopping?  
            4. Virality Potential – Would people retweet or share it?  
            5. Format – Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

            Auto-reject if:
            - It's written in question-answer format (e.g., "Why did..." or "What happens when...")
            - It exceeds 280 characters
            - It reads like a traditional setup-punchline joke
            - Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)

            ### Respond ONLY in structured format:
            - evaluation: "approved" or "needs_improvement"  
            - feedback: One paragraph explaining the strengths and weaknesses
            """
        ),
    ],
    input_variables=["post"]
)

optimizationPrompt = ChatPromptTemplate(
    messages=[
        ("system","You punch up tweets for virality and humor based on given feedback."),
        ("human","""
            Improve the tweet based on this feedback:
            "{feedback}"

            Topic: "{topic}"
            Original Tweet:
            {post}

            Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
            """
        ),
    ],
    input_variables=["post", "feedback", "topic"]
)

## LLM Chains
generationChain = generationPrompt | generationModel | StrOutputParser()
evaluationChain = evaluationPrompt | structuredEvaluationModel
optimizationChain = optimizationPrompt | optimizationModel | StrOutputParser()

# Graph State
class PostGenerator(TypedDict):
    topic: str
    post: str
    evaluation: Literal["approved","needs_improvement"]
    feedback: str
    iteration: int = 0
    maxIteration: int
    postHistory: Annotated[list[str],operator.add]
    feedbackHistory: Annotated[list[str],operator.add]

# Graph Node Operations and Condition Checkers
def generatePost(state:PostGenerator)->PostGenerator:
    generationResults = generationChain.invoke({
        "topic": state["topic"]
    })
    return {
        "post": generationResults,
        "postHistory": [generationResults]
    }

def evaluatePost(state:PostGenerator)->PostGenerator:
    evaluationResult = evaluationChain.invoke({
        "post": state["post"]
    })
    return {
        "evaluation": evaluationResult.evaluation,
        "feedback": evaluationResult.feedback,
        "feedbackHistory": [evaluationResult.feedback],
    }

def optimizePost(state:PostGenerator)->PostGenerator:
    optimizationResult = optimizationChain.invoke({
        "post": state["post"],
        "feedback": state["feedback"],
        "topic": state["topic"]
    })
    return {
        "post": optimizationResult,
        "iteration": state["iteration"] + 1,
        "postHistory": [optimizationResult]
    }

def routeEvaluation(state:PostGenerator)->Literal['approved','needs_improvement']:
    if state["evaluation"] == 'approved' or state["iteration"] >= state["maxIteration"]: 
        return "approved"
    else:
        return "needs_improvement"

# Graph Building
graph_builder = StateGraph(PostGenerator)

# Nodes of Graph
graph_builder.add_node("generate_post", generatePost)
graph_builder.add_node("evaluate_post", evaluatePost)
graph_builder.add_node("optimize_post", optimizePost)

# Edges of Graph
graph_builder.add_edge(START, "generate_post")
graph_builder.add_edge("generate_post", "evaluate_post")
graph_builder.add_conditional_edges(
    "evaluate_post",
    routeEvaluation,
    {
        "approved": END,
        "needs_improvement": "optimize_post"
    }
)
graph_builder.add_edge("optimize_post", "evaluate_post")

# Graph Compiling
graph = graph_builder.compile()

# Graph View
print(graph.get_graph().print_ascii())

# Graph Invoking
response = graph.invoke({
    "topic": "AI in Healthcare",
    "iteration": 1,
    "maxIteration": 3
})

print("Final Post:",response['post'])
print("Iteration:",response['iteration'])
print("Post History:", response['postHistory'])