from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from marshmallow.fields import List
from src.pdf_loader import all_docs

all_chunks = []
def chunking(all_docs:List[Document]) -> list[Document]:
    # Chunk（切块）→ Embedding（向量化）→ Vector Store（入库）→ Retriever（检索）→ LLM（生成回答）
    # Chunk: pages -> chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap= 150)
    chunks = splitter.split_documents(all_docs)
    all_chunks.extend(chunks)
    return all_chunks