from fastapi import FastAPI, UploadFile, File
import shutil
import os

from backend.app.utils import extract_text_from_pdf, chunk_text
from backend.app.rag import create_vector_store, build_rag_chain
from backend.app.agents import build_agent

# import os
# from dotenv import load_dotenv

# load_dotenv()

# print("KEY:", os.getenv("OPENAI_API_KEY"))

app = FastAPI()

rag_chain = None
agent = None

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    path = f"temp_{file.filename}"
    
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_from_pdf(path)
    chunks = chunk_text(text)

    create_vector_store(chunks)

    global rag_chain, agent
    rag_chain = build_rag_chain()
    agent = build_agent()

    os.remove(path)

    return {"message": "File processed successfully"}

@app.post("/ask/")
async def ask_question(query: str):
    if agent is None:
        return {"error": "Upload documents first"}

    response = agent.run(query)
    return {"response": response}

@app.post("/rag/")
async def rag_query(query: str):
    if rag_chain is None:
        return {"error": "Upload documents first"}

    result = rag_chain(query)
    
    return {
        "answer": result["result"],
        "sources": [doc.page_content[:200] for doc in result["source_documents"]]
    }