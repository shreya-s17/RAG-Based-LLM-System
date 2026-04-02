# RAG-Based-LLM-System

## 🧠 AI Research Assistant (RAG + Multi-Agent System)

An end-to-end **Retrieval-Augmented Generation (RAG)** + **Multi-Agent AI System** that allows users to upload documents (PDFs) and interact with them using natural language queries.

Built with a **Planner → Executor → Critic architecture**, this system ensures grounded, reliable, and explainable AI responses with **citations and hallucination detection**.

---

## 🚀 Features

### 🔍 RAG Pipeline
- Document ingestion (PDF)
- Text chunking & embedding
- Vector search (FAISS)
- Context-aware question answering

### 🤖 Multi-Agent Architecture
- **Planner Agent** → breaks query into steps  
- **Executor Agent** → retrieves context + generates answer  
- **Critic Agent** → verifies correctness and improves output  

### 📚 Grounded Responses
- Answers generated strictly from retrieved documents  
- Source attribution with `[SOURCE #]` citations  

### 📊 Hallucination Detection
- Semantic similarity scoring using embeddings  
- Confidence score for every response  

### ⚙️ Tool Integration
- Document search tool (RAG)
- Python execution tool (extensible)

---

## 🏗️ System Architecture


User Query
↓
Planner Agent
↓
Executor Agent
↓
RAG Retrieval (Vector DB)
↓
LLM Response
↓
Critic Agent (Verification + Scoring)
↓
Final Answer + Confidence + Sources


---

## 🧰 Tech Stack

### 🧠 AI / LLM
- OpenAI GPT Models (via LangChain)

### 🔗 Frameworks
- LangChain (RAG + Agents)

### 🗂️ Vector Database
- FAISS (local vector store)

### ⚙️ Backend
- FastAPI

### 💻 Frontend
- Streamlit

### 📊 Embeddings
- OpenAI Embeddings
- SentenceTransformers (for evaluation)

---

## 📁 Project Structure


llm-research-assistant/
│
├── backend/
│ ├── app/
│ │ ├── main.py # FastAPI API
│ │ ├── rag.py # RAG pipeline + citations
│ │ ├── agents.py # Multi-agent system
│ │ ├── utils.py # chunking + evaluation
│ │ └── config.py
│ ├── requirements.txt
│
├── frontend/
│ └── streamlit_app.py # UI
│
├── ingestion/
│ └── ingest.py # data ingestion
│
├── Dockerfile
├── README.md
└── .env


---

## ⚙️ Installation

### 1. Clone Repo
```bash
git clone https://github.com/your-username/llm-research-assistant.git
cd llm-research-assistant
2. Create Virtual Environment
python -m venv .venv
source .venv/bin/activate
3. Install Dependencies
pip install -r backend/requirements.txt
4. Add API Key

Create .env file:

OPENAI_API_KEY=your_api_key_here
▶️ Running the App
Start Backend
cd backend
uvicorn app.main:app --reload
Start Frontend
cd frontend
streamlit run streamlit_app.py
🧪 API Endpoints
Upload Document
POST /upload/
Ask via RAG
POST /rag/?query=your_question
Run Multi-Agent System
POST /agent/?query=your_question
🧠 Example Query
"Summarize this paper and create a LinkedIn post"
Output:
Step-by-step plan
Answer grounded in documents
Confidence score
Source citations
📊 Evaluation & Reliability
✅ Faithfulness Score
Uses embedding similarity to measure grounding
⚠️ Confidence Levels
High → Reliable
Medium → Needs verification
Low → Likely hallucination
🔐 Safety Considerations
Avoids answering outside provided context
Returns "I don’t know" if information unavailable
(Recommended) Sandbox Python execution for production
🚀 Future Improvements
Hybrid search (keyword + semantic)
Reranking models
Long-term memory
Multi-agent parallel execution
Production deployment (AWS / Render)
UI upgrade (React + streaming)
📌 Resume Description

Built an end-to-end Retrieval-Augmented Generation (RAG) system with a multi-agent architecture (Planner, Executor, Critic), incorporating citation grounding, hallucination detection, and confidence scoring for reliable AI responses.

⭐ Why This Project Matters
Demonstrates real-world AI system design
Covers LLMs + RAG + Agents
Includes evaluation + reliability (rare skill)
Aligns with industry-level AI engineering
