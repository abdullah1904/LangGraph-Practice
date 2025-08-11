from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import operator

load_dotenv()

model = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct", 
    max_tokens=1024,
    temperature=0.5,  
)

class EssayEvaluatorSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score for the essay on a scale of 1 to 10", ge=0, le=10)

structuredModel = model.with_structured_output(EssayEvaluatorSchema)

clarityOfThoughtPrompt = PromptTemplate(
    input_variables=["essayText"],
    template="Evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10 \n {essayText}"
)

depthOfAnalysisPrompt = PromptTemplate(
    input_variables=["essayText"],
    template="Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10 \n {essayText}"
)

languagePrompt = PromptTemplate(
    input_variables=['essayText'],
    template="Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {essayText}"
)

finalEvaluationPrompt = PromptTemplate(
    input_variables=["clarityOfThoughtFeedback", "depthOfAnalysisFeedback", "languageFeedback"],
    template="Based on the following feedbacks create a summarized feedback \n Clarity of Thought: {clarityOfThoughtFeedback} \n Depth of Analysis: {depthOfAnalysisFeedback} \n Language Quality: {languageFeedback}"
)

clarityOfThoughtChain = clarityOfThoughtPrompt | structuredModel
depthOfAnalysisChain = depthOfAnalysisPrompt | structuredModel 
languageChain = languagePrompt | structuredModel
finalEvaluationChain = finalEvaluationPrompt | model | StrOutputParser()

class EssayEvaluatorState(TypedDict):
    essayText: str
    clarityOfThoughtFeedback: str
    depthOfAnalysisFeedback: str
    languageFeedback: str
    summarizeFeedback: str
    individualScores: Annotated[list[int], operator.add]
    finalScore: float

graph_builder = StateGraph(EssayEvaluatorState)

def evaluateClarityOfThought(state: EssayEvaluatorState):
    clarityOfThoughtResult = clarityOfThoughtChain.invoke({"essayText": state["essayText"]})
    return {
        "clarityOfThoughtFeedback": clarityOfThoughtResult.feedback,
        "individualScores": [clarityOfThoughtResult.score],
    }

def evaluateDepthOfAnalysis(state: EssayEvaluatorState):
    depthOfAnalysisResult = depthOfAnalysisChain.invoke({"essayText": state["essayText"]})
    return {
        "depthOfAnalysisFeedback": depthOfAnalysisResult.feedback,
        "individualScores": [depthOfAnalysisResult.score],
    }

def evaluateLanguage(state: EssayEvaluatorState):
    languageResult = languageChain.invoke({"essayText": state["essayText"]})
    return {
        "languageFeedback": languageResult.feedback,
        "individualScores": [languageResult.score],
    }

def finalEvaluation(state: EssayEvaluatorState):
    summarizeFeedback = finalEvaluationChain.invoke({
        "clarityOfThoughtFeedback": state["clarityOfThoughtFeedback"],
        "depthOfAnalysisFeedback": state["depthOfAnalysisFeedback"],
        "languageFeedback": state["languageFeedback"],
    })
    finalScore = sum(state["individualScores"]) / len(state["individualScores"])
    return {
        "summarizeFeedback": summarizeFeedback,
        "finalScore": finalScore
    }

# Nodes of Graph
graph_builder.add_node("evaluate_clarity_of_thought", evaluateClarityOfThought)
graph_builder.add_node("evaluate_depth_of_analysis", evaluateDepthOfAnalysis)
graph_builder.add_node("evaluate_language", evaluateLanguage)
graph_builder.add_node("final_evaluation", finalEvaluation)

# Edges of Graph
graph_builder.add_edge(START, "evaluate_clarity_of_thought")
graph_builder.add_edge(START, "evaluate_depth_of_analysis")
graph_builder.add_edge(START, "evaluate_language")

graph_builder.add_edge("evaluate_clarity_of_thought", "final_evaluation")
graph_builder.add_edge("evaluate_depth_of_analysis", "final_evaluation")
graph_builder.add_edge("evaluate_language", "final_evaluation")

graph_builder.add_edge("final_evaluation", END)

graph = graph_builder.compile()

print(graph.get_graph().print_ascii())

essay = """
Pakistan in Tech

Pakistan is gradually emerging as a promising player in the global technology landscape. Over the past decade, the country has witnessed a significant increase in internet penetration, smartphone usage, and access to digital services. With a population exceeding 240 million, a large portion of which is young and tech-savvy, Pakistan holds enormous potential for technological growth. This youthful demographic, coupled with growing entrepreneurial enthusiasm, has paved the way for innovation in software development, e-commerce, fintech, health tech, and other digital domains.

The government of Pakistan has taken various initiatives to promote the IT sector, such as offering tax exemptions for IT exports, establishing Special Technology Zones (STZs), and encouraging foreign investment. Moreover, programs like the "DigiSkills" training initiative and "National Incubation Centers" aim to equip youth with digital literacy and entrepreneurial skills. These measures are helping to develop a skilled workforce capable of competing in global markets.

In the private sector, startups are playing a crucial role in transforming Pakistan's tech ecosystem. Platforms like Daraz, Airlift, Bykea, and Bazaar have shown how technology can revolutionize industries ranging from retail to logistics. The fintech sector has seen rapid growth with mobile banking solutions like Easypaisa and JazzCash, making financial services accessible to millions, especially in rural areas. The software export industry is also thriving, with Pakistani developers and IT companies serving clients worldwide, particularly in North America and Europe.

However, challenges remain. Limited infrastructure in remote areas, inconsistent internet speeds, and a shortage of advanced research facilities hinder rapid progress. Moreover, brain drain, where skilled professionals migrate for better opportunities abroad, continues to be a pressing concern. Cybersecurity awareness and regulations also need to be strengthened to ensure the safety of users and businesses in the digital space.

Despite these obstacles, the future looks promising. With investments in education, infrastructure, and innovation-friendly policies, Pakistan can position itself as a strong technology hub in South Asia. The global shift toward remote work, artificial intelligence, and digital transformation presents new opportunities for Pakistani talent to shine on the international stage. If harnessed effectively, technology can not only drive economic growth but also improve governance, education, healthcare, and overall quality of life for millions of Pakistanis.

Pakistan's journey in technology is still in its early stages, but the momentum is undeniable. With its youthful population, expanding digital economy, and growing pool of skilled professionals, the country has the potential to be a significant contributor to the global tech industry, shaping a brighter, more connected future.
"""

response = graph.invoke({
    "essayText": essay,
})

print(response['summarizeFeedback'])