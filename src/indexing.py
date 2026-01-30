from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.chunking import all_chunks

# HuggingFaceEmbeddings 知道怎么把 text -> vector（维度固定）
# Chroma.from_documents 调 embedding.embed_documents() 把向量 + metadata + text 存起来
# Embedding
model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# Vector Store
vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=model,
    persist_directory="chroma_db"
)

def format_context(docs) -> str:
    blocks = []
    for doc in docs:
        text = doc.page_content.strip()[:1000]
        source = Path(doc.metadata['source']).name
        page = doc.metadata.get("page")
        blocks.append(f"{source}:{page}\n{text}\n")
    return "\n\n".join(blocks)