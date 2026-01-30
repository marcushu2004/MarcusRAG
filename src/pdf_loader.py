from pathlib import Path
from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader

all_docs = []
def pdf_loader() -> list[Document]:
    # check the path
    print("Please save your documents that you want to read in 'data/pdfs' path")
    print("请将你要读取的文本存入'data/pdf'路径下")
    p = Path("../data/pdfs")
    if p.exists() != False:
        pass
    else:
        Path("../data/pdfs").mkdir()

    # read files
    print("Loading PDFs:\n")
    for file in p.glob("*.pdf"):
        print(f"Loading: {file.name}")
        loader = PyPDFLoader(str(file))
        docs = loader.load()
        all_docs.extend(docs)

        print(f"{len(docs)} documents loaded\n")
        print("Preview:")
        print(docs[0].page_content[:200])
        print("Metadata:")
        print(docs[0].metadata)
    return all_docs
