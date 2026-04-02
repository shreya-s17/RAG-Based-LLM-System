from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

import os
import numpy as np
from dotenv import load_dotenv
from backend.app.utils import compute_faithfulness

load_dotenv()

FAISS_PATH = "faiss_index"

# ✅ Create + SAVE vector store
def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    docs = [Document(page_content=chunk) for chunk in chunks]

    vector_db = FAISS.from_documents(docs, embeddings)

    # 🔥 CRITICAL: persist to disk
    vector_db.save_local(FAISS_PATH)

    return vector_db


# ✅ Create + SAVE vector store with BM25 index
VECTOR_DB = None
BM25_INDEX = None
DOCS = []

def create_vector_store(chunks):
    global VECTOR_DB, BM25_INDEX, DOCS

    embeddings = OpenAIEmbeddings()
    VECTOR_DB = FAISS.from_texts(chunks, embeddings)

    # BM25 setup
    tokenized_docs = [doc.split(" ") for doc in chunks]
    BM25_INDEX = BM25Okapi(tokenized_docs)

    DOCS = chunks

    return VECTOR_DB



# ✅ Load vector store (for Render restarts)
def load_vector_store():
    if not os.path.exists(FAISS_PATH):
        return None

    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    return FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )



# ✅ Get retriever safely
def get_retriever():
    vector_db = load_vector_store()

    if vector_db is None:
        raise ValueError("Vector store not found. Upload document first.")

    return vector_db.as_retriever(search_kwargs={"k": 4})


# ✅ Get Hybrid retriever safely
def hybrid_retrieve(query, k=5):
    # Semantic search
    semantic_docs = VECTOR_DB.similarity_search(query, k=k)

    # Keyword search
    tokenized_query = query.split(" ")
    bm25_scores = BM25_INDEX.get_scores(tokenized_query)

    top_bm25_indices = np.argsort(bm25_scores)[-k:]
    keyword_docs = [DOCS[i] for i in top_bm25_indices]

    # Combine results
    combined = list(set([doc.page_content for doc in semantic_docs] + keyword_docs))

    return combined


# ✅ Build RAG chain
def build_rag_chain():
    llm = ChatOpenAI(
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    retriever = get_retriever()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )


# ✅ Build RAG with citations and faithfulness scoring
def critic_with_scoring(query, answer, docs):
    score = compute_faithfulness(answer, docs)

    if score < 0.3:
        return {
            "final_answer": "⚠️ Low confidence: Answer may not be grounded in documents.",
            "confidence": score
        }

    elif score < 0.6:
        return {
            "final_answer": answer + "\n\n⚠️ Partial confidence. Verify sources.",
            "confidence": score
        }

    else:
        return {
            "final_answer": answer,
            "confidence": score
        }