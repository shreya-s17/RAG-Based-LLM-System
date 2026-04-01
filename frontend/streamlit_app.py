import streamlit as st
import requests

# API_URL = "https://rag-based-llm-system-6wde.onrender.com"
API_URL = "http://localhost:8000"  # Use this for local testing

st.title("🧠 AI Research Assistant")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
    }

    with st.spinner("Uploading and processing..."):
        response = requests.post(f"{API_URL}/upload/", files=files)

    if response.status_code == 200:
        st.success("File uploaded!")
    else:
        st.error(response.text)

query = st.text_input("Ask a question")

if st.button("Ask Agent"):
    res = requests.post(f"{API_URL}/ask/", json={"query": query})
    
    if res.status_code == 200:
        st.write(res.json())
    else:
        st.error(res.text)

if st.button("Ask RAG"):
    res = requests.post(f"{API_URL}/rag/", json={"query": query})

    if res.status_code == 200:
        st.write(res.json())
    else:
        st.error(res.text)

if st.button("Run Multi-Agent"):
    res = requests.post(f"{API_URL}/agent/", json={"query": query})
    data = res.json()

    if res.status_code != 200:
        st.error(f"API Error: {res.text}")
    else:
        data = res.json()

        st.subheader("🧠 Plan")
        st.write(data.get("plan", "No plan returned"))

        st.subheader("⚙️ Draft")
        st.write(data.get("draft", "No draft returned"))

        st.subheader("✅ Final Answer")
        st.write(data.get("final", "No final answer returned"))