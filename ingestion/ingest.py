import sys
# ingest.py
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(path):
    text = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text.append({"page": i+1, "text": page.extract_text() or ""})
    return text

def chunk_document(text_blocks, chunk_size=1000, chunk_overlap=200):
    join_text = "\n".join([b["text"] for b in text_blocks])
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(join_text)
    return chunks

# usage
# pages = extract_text_from_pdf("sample.pdf")
# chunks = chunk_document(pages)