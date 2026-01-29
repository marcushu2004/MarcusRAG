from pathlib import Path
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.testing.suite.test_reflection import metadata

# from langchain.text_splitter import RecursiveCharacterTextSplitter

# check the path
print("Please save your documents that you want to read in 'data/pdfs' path")
print("请将你要读取的文本存入'data/pdf'路径下")
p = Path("data/pdfs")
if p.exists() != False:
    pass
else:
    Path("data/pdfs").mkdir()

# read files
print("Loading PDFs:\n")
for file in p.glob("*.pdf"):
    print(f"Loading: {file.name}")

    loader = PyPDFLoader(str(file))
    docs = loader.load()

    print(f"{len(docs)} documents loaded\n")
    print("Preview:")
    print(docs[0].page_content[:200])
    print("Metadata:")
    print(docs[0].metadata)

# Chunk（切块）→ Embedding（向量化）→ Vector Store（入库）→ Retriever（检索）→ LLM（生成回答）
# Chunk: pages -> chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap= 150)
chunks = splitter.split_documents(docs)
# Test data full or broken
# print(len(docs))
# print(len(chunks))
# print(chunks[0].page_content)
# print(chunks[0].metadata)
# print(chunks[1].page_content[-200:])
# print(chunks[2].page_content[:200])

# HuggingFaceEmbeddings 知道怎么把 text -> vector（维度固定）
# Chroma.from_documents 调 embedding.embed_documents() 把向量 + metadata + text 存起来
# Embedding
model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# Vector Store
vectorstore = Chroma.from_documents(
    documents=chunks,
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

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":4, "fetch_k":20})
question = "How is the policy network trained in AlphaGo?"
docs = retriever.invoke(question)
context = format_context(docs)
prompt_str = f"You are a assistant. Answer the questions ONLY using the context. If you can not find the answer. Use citations (page numbers) in Sources.,{context},{question}"


# for docs in results:
#     print(docs.metadata["page"])
#     print(docs.metadata["source"])
#     print(docs.page_content[:200])

llm = ChatOllama(model="qwen2.5:3b")
print(len(docs))
resp = llm.invoke(prompt_str)

print([d.metadata.get("page") for d in docs])
print(context[:400])