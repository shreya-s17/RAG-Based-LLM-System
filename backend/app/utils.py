import pdfplumber
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file_path):
    text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_text(text)


def compute_faithfulness(answer, docs):
    answer_embedding = model.encode(answer, convert_to_tensor=True)

    scores = []

    for doc in docs:
        doc_embedding = model.encode(doc.page_content, convert_to_tensor=True)
        similarity = util.cos_sim(answer_embedding, doc_embedding)
        scores.append(similarity.item())

    max_score = max(scores) if scores else 0

    return max_score