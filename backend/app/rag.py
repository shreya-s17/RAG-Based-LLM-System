from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


VECTOR_DB = None

def create_vector_store(chunks):
    global VECTOR_DB
    print(os.getenv("OPENAI_API_KEY")) 
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    docs = [Document(page_content=chunk) for chunk in chunks]
    VECTOR_DB = FAISS.from_documents(docs, embeddings)
    return VECTOR_DB

def get_retriever():
    return VECTOR_DB.as_retriever(search_kwargs={"k": 4})

def build_rag_chain():
    llm = ChatOpenAI(temperature=0)
    retriever = get_retriever()
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )