from sqlalchemy.orm import Session
from sqlalchemy import text
from database import get_db
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import datetime
import time
load_dotenv()

# # API keys for RAG services
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY="sk--wgT5Os0yh65yiaLY39ycJii98y9zQ6Y4alSi4dQ48T3BlbkFJHZHitsUUCDO48X3MDSWMFdtpeotEy71XvyQMBpf8cA"
ANTHROPIC_API_KEY="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"


def initialize_rag(csv_file: str):
    df = pd.read_csv(csv_file, encoding='latin1')
    documents = [Document(page_content=row.to_json(), metadata=row.to_dict()) for _, row in df.iterrows()]
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()
    chat_model = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
    rag_chain = ConversationalRetrievalChain.from_llm(chat_model, retriever=retriever)
    return rag_chain

rag_chain = initialize_rag("data.csv")  # Replace with your CSV file path

class QueryRequest(BaseModel):
    prompt: str

def query_sql(user_query: str):
    with get_db() as db:
        sql_query = f"SELECT * FROM content WHERE text LIKE '%{user_query}%'"
        result = db.execute(text(sql_query)).fetchall()
        return [row['text'] for row in result]

# RAG Query
metrics = {
    "total_queries": 0,
    "total_query_time": 0,
    "relevancy_scores": [],
}

def embed_text(text):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    embeddings = embeddings.embed_query(text)
    return embeddings

def calculate_relevancy_score(question, answer, faiss_score):
    normalized_faiss_score = min(max(faiss_score / 100, 0), 1)  # Adjust based on your scoring system

    question_embedding = embed_text(question)
    answer_embedding = embed_text(answer)
    if question_embedding is None or answer_embedding is None:
        return 0.0  
    cosine_sim = cosine_similarity([question_embedding], [answer_embedding])[0][0]
    relevancy_score = (normalized_faiss_score + cosine_sim) / 2
    return relevancy_score

# Function to simulate logging engagement (for demonstration)
def log_engagement(request_time, relevancy_score):
    # Increment total queries
    metrics["total_queries"] += 1
    # Track total query time
    metrics["total_query_time"] += request_time
    # Add relevancy score to the list
    metrics["relevancy_scores"].append(relevancy_score)
    print(f"Metrics Updated: Total Queries = {metrics['total_queries']}, Avg Query Time = {metrics['total_query_time'] / metrics['total_queries']}")

def query_rag(request: QueryRequest):
    now = datetime.datetime.now()
    start_time = time.time()
    try:
        response = rag_chain({"question": request, "chat_history": []})
        faiss_scores = response.get('faiss_scores', [0])
        end_time = time.time()
        request_time = end_time - start_time
        answer = response.get('answer', 'No answer found.')
        relevancy_score = calculate_relevancy_score(request, answer, faiss_scores[0])
        log_engagement(request_time, relevancy_score)
        return {"response": answer, 
                "query_time": request_time,
                "relevancy_score": relevancy_score,
                "total_queries": metrics['total_queries'],
                "avg_query_time": metrics['total_query_time'] / metrics['total_queries']
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Hybrid Query
def hybrid_query(user_query: str, use_rag: bool):
    if use_rag:
        return query_rag(user_query)
    else:
        return query_sql(user_query)
