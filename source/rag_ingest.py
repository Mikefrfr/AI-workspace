import os
import tiktoken
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_BASE_DIR = "./chroma_db"
tokenizer       = tiktoken.get_encoding("cl100k_base")

def token_len(text):
    return len(tokenizer.encode(text))


def read_file(path: str) -> str:
    ext = Path(path).suffix.lower()

    if ext in (".txt", ".md"):
        return open(path, encoding="utf-8", errors="ignore").read()

    elif ext == ".pdf":
        import pdfplumber
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text

    elif ext in (".docx", ".doc"):
        import docx
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    elif ext == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        return df.to_string()

    elif ext in (".xlsx", ".xls"):
        import pandas as pd
        df = pd.read_excel(path)
        return df.to_string()

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def ingest_file(path: str, collection_name: str = "default") -> dict:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print(f"[ingest] reading {path}...")
    text = read_file(path)

    if not text.strip():
        raise ValueError("File appears to be empty or unreadable.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=token_len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.create_documents(
        [text],
        metadatas=[{"source": Path(path).name}] * 1
    )
    print(f"[ingest] {len(chunks)} chunks created")

    chroma_path = os.path.join(CHROMA_BASE_DIR, collection_name)
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=chroma_path
    )
    print(f"[ingest] stored in {chroma_path}")

    return {
        "chunks":      len(chunks),
        "collection":  collection_name,
        "source":      Path(path).name,
        "chroma_path": chroma_path,
    }