import streamlit as st
import requests

API_URL = "https://rag-based-llm-system-6wde.onrender.com"

st.title("🧠 AI Research Assistant")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(f"{API_URL}/upload/", files={"file": uploaded_file})
    st.success("File uploaded!")

query = st.text_input("Ask a question")

if st.button("Ask Agent"):
    res = requests.post(f"{API_URL}/ask/", params={"query": query})
    st.write(res.json())

if st.button("Ask RAG"):
    res = requests.post(f"{API_URL}/rag/", params={"query": query})
    st.write(res.json())