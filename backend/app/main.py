from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os

from backend.app.utils import extract_text_from_pdf, chunk_text
from backend.app.rag import create_vector_store, build_rag_chain, create_vector_store_with_bm25
from backend.app.agents import build_agent, run_multi_agent_with_citations, run_multi_agent, run_rag_with_citations_hybrid_retrieve_and_reranking, run_rag_with_memory
from backend.app.memory import ConversationMemory

memory = ConversationMemory()
app = FastAPI()

# ✅ CORS (required for Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Request schema
class QueryRequest(BaseModel):
    query: str

# ✅ Globals (rebuilt from disk when needed)
rag_chain = None
agent = None

# ✅ Health check
@app.get("/")
def root():
    return {"message": "API is running"}

# ✅ Upload and process PDF
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        path = f"temp_{file.filename}"

        # Save uploaded file
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract + chunk
        text = extract_text_from_pdf(path)
        chunks = chunk_text(text)

        # Create and persist vector store
        create_vector_store(chunks)  # must internally save FAISS
        create_vector_store_with_bm25(chunks)  # also create BM25 index

        # Build chains
        global rag_chain, agent
        rag_chain = build_rag_chain()
        agent = build_agent()

        # Cleanup
        os.remove(path)

        return {"message": "File processed successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ✅ Ensure models are loaded (important for Render restart)
def ensure_models_loaded():
    global rag_chain, agent

    if rag_chain is None or agent is None:
        try:
            rag_chain = build_rag_chain()
            agent = build_agent()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="No vector store found. Upload a document first."
            )


# ✅ Agent endpoint
@app.post("/ask/")
async def ask_question(req: QueryRequest):
    ensure_models_loaded()

    try:
        response = agent.run(req.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ✅ RAG endpoint
@app.post("/rag/")
async def rag_query(req: QueryRequest):
    ensure_models_loaded()

    try:
        result = rag_chain(req.query)

        return {
            "answer": result["result"],
            "sources": [
                doc.page_content[:200] for doc in result["source_documents"]
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# ✅ Multi-Agent endpoint
@app.post("/agent/")
async def multi_agent(req: QueryRequest):
    try:
        print("Received query:", req.query)

        result = run_multi_agent(req.query)

        print("Returning:", result)

        return result

    except Exception as e:
        print("ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    

# ✅ Multi-Agent with citations endpoint
@app.post("/agent_with_citations/")
async def multi_agent_with_citations(req: QueryRequest):
    try:
        result = run_multi_agent_with_citations(req.query)

        return result

    except Exception as e:
        print("ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    

# ✅ Multi-Agent with citations - hybrid retrieval and reranking endpoint
@app.post("/agent_with_citations_hybrid_retrieval_and_reranking/")
async def multi_agent_with_citations_hybrid_retrieval_and_reranking(req: QueryRequest):
    try:
        print("Received query:", req.query)

        result = run_rag_with_citations_hybrid_retrieve_and_reranking(req.query)

        print("Returning:", result)

        return result

    except Exception as e:
        print("ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    

# ✅ RAG with memory endpoint
@app.post("/chat/")
async def chat(req: QueryRequest):
    response = run_rag_with_memory(req.query, memory)
    return {"response": response}