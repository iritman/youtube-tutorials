# -*- coding: utf-8 -*-

# !pip install -q langchain langchain-openai langchain-community langgraph faiss-cpu

from google.colab import userdata
import os

OPENROUTER_API_KEY = userdata.get("OPENROUTER_API_KEY")

BASE_URL = "https://openrouter.ai/api/v1"

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

model="openai/gpt-oss-20b:free"

llm = ChatOpenAI(
    model=model,
    temperature=0,
    base_url=BASE_URL,
    api_key=OPENROUTER_API_KEY
)

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url=BASE_URL,
    api_key=OPENROUTER_API_KEY
)

class AgentState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    needs_retrieval: bool

sample_texts = [
    "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
    "RAG stands for Retrieval-Augmented Generation and combines retrieval with generation.",
    "A vector database stores high-dimensional embeddings for semantic search.",
    "Agentic AI systems can reason, decide, retrieve information, and generate responses dynamically."
]

documents = [Document(page_content=text) for text in sample_texts]

vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(k=3)

def decide_retrieval(state: AgentState) -> AgentState:
    """
    Decide if retrieval is needed using an LLM.
    The LLM is asked to classify if the question is related to
    LangGraph, RAG, Vector DB, or Agentic AI.
    """
    question = state["question"]

    prompt = f"""
You are an AI assistant. Determine if the following question
is related to any of these topics: LangGraph, RAG, Vector DB, Agentic AI.

Question: "{question}"

Answer "YES" if retrieval is needed or "NO" if not.
"""

    response = llm.invoke(prompt)
    answer_text = response.content.strip().lower()

    needs_retrieval = answer_text.startswith("yes")
    print(needs_retrieval)
    return {**state, "needs_retrieval": needs_retrieval}

# def decide_retrieval(state: AgentState) -> AgentState:
#     question = state["question"]

#     retrieval_keywords = ["what", "how", "explain", "describe", "tell me"]

#     needs_retrieval = any(
#         keyword in question.lower() for keyword in retrieval_keywords
#     )

#     return {**state, "needs_retrieval": needs_retrieval}

def retrieve_documents(state: AgentState) -> AgentState:
    """
    Retrieve relevant documents based on the question
    """
    question = state["question"]
    documents = retriever.invoke(question)

    return {**state, "documents": documents}

def generate_answer(state: AgentState) -> AgentState:
    """
    Generate an answer using the retrieved documents or direct response
    """
    question = state["question"]
    documents = state.get("documents", [])

    if documents:
        # RAG approach: use documents as context
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = f"""Based on the following context, answer the question:

                      Context:
                      {context}

                      Question: {question}

                      Answer:"""
    else:
        # Direct response without retrieval
        prompt = f"Answer the following question: {question}"

    response = llm.invoke(prompt)
    answer = response.content

    return {**state, "answer": answer}

def should_retrieve(state: AgentState) -> str:
    """
    Determine the next step based on retrieval decision
    """
    if state["needs_retrieval"]:
        return "retrieve"
    else:
        return "generate"

# Create the state graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("decide", decide_retrieval)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)

# Set entry point
workflow.set_entry_point("decide")

# Add conditional edges
workflow.add_conditional_edges(
    "decide",
    should_retrieve,
    {
        "retrieve": "retrieve",
        "generate": "generate"
    }
)

# Add edges
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the graph
app = workflow.compile()
app

def ask_question(question: str):
    """
    Helper function to ask a question and get an answer
    """
    initial_state = {
        "question": question,
        "documents": [],
        "answer": "",
        "needs_retrieval": False
    }

    result = app.invoke(initial_state)
    return result

# Test with a question that should trigger retrieval
question1 = "What is LangGraph?"
result1 = ask_question(question1)
result1

# Test with another question
question2 = "How does RAG work?"
result2 = ask_question(question2)

print(f"Question: {question2}")
print(f"Retrieved documents: {len(result2['documents'])}")
print(f"Answer: {result2['answer']}")
print("\n" + "="*50 + "\n")

# Test with another question
question3 = "Where is the capital of France?"
result3 = ask_question(question3)

print(f"Question: {question3}")
print(f"Retrieved documents: {len(result3['documents'])}")
print(f"Answer: {result3['answer']}")
print("\n" + "="*50 + "\n")