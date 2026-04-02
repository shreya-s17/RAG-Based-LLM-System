import streamlit as st
import requests

# API_URL = "https://rag-based-llm-system-6wde.onrender.com"
API_URL = "http://localhost:8000"  # Use this for local testing

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("🧠 AI Research Assistant")

# =========================
# 🔹 SESSION MEMORY (CHAT)
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# 🔹 CHAT DISPLAY
# =========================
st.subheader("💬 Chat (Memory Enabled)")

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

user_input = st.chat_input("Ask something about your documents...")

if user_input:
    # Store user input
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Call backend memory endpoint
    response = requests.post(
        f"{API_URL}/chat/",
        json={"query": user_input}   # ✅ FIX
    )

    if response.status_code == 200:
        answer = response.json().get("response", "Error")

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })
    else:
        st.error(response.text)

# =========================
# 🔹 CLEAR MEMORY
# =========================
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.success("Chat cleared!")



# =========================
# 🔹 FILE UPLOAD 
# =========================
st.subheader("📄 Upload Document")

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

# =========================
# 🔹 MANUAL QUERY SECTION 
# =========================
st.subheader("🔍 Manual Query (Debug / Testing)")

query = st.text_input("Ask a question")

# -------------------------
# BASIC AGENT
# -------------------------

if st.button("Ask Agent"):
    res = requests.post(f"{API_URL}/ask/", json={"query": query})
    
    if res.status_code == 200:
        st.write(res.json())
    else:
        st.error(res.text)

# -------------------------
# BASIC RAG
# -------------------------

if st.button("Ask RAG"):
    res = requests.post(f"{API_URL}/rag/", json={"query": query})

    if res.status_code == 200:
        st.write(res.json())
    else:
        st.error(res.text)

# -------------------------
# MULTI AGENT
# -------------------------

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


# -------------------------
# MULTI AGENT + CITATIONS
# -------------------------

if st.button("Run Multi-Agent-With-Citations"):
    res = requests.post(f"{API_URL}/agent_with_citations/", json={"query": query})
    data = res.json()

    if res.status_code != 200:
        st.error(f"API Error: {res.text}")
    else:
        data = res.json()

    st.subheader("🧠 Plan")
    st.write(data["plan"])

    st.subheader("✅ Final Answer")
    st.write(data["answer"])

    st.subheader("📊 Confidence Score")
    st.write(data["confidence"])

    st.subheader("📚 Sources")
    for s in data["sources"]:
        st.write(f"Source {s['id']}: {s['content']}")


# -------------------------
# HYBRID + RERANKING
# -------------------------

if st.button("Run Multi-Agent-With-Citations-Hybrid-Retrieval-And-Reranking"):
    res = requests.post(f"{API_URL}/agent_with_citations_hybrid_retrieval_and_reranking/", json={"query": query})
    data = res.json()

    if res.status_code != 200:
        st.error(f"API Error: {res.text}")
    else:
        data = res.json()

    st.subheader("✅ Final Answer")
    st.write(data["answer"])

    st.subheader("📊 Raw Document")
    st.write(data["raw_docs"])

    st.subheader("📚 Sources")
    for s in data["sources"]:
        st.write(f"Source {s['id']}: {s['content']}")