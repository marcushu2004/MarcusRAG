from pdf_loader import docs
from llm import llm
from retrieval import context, prompt_str

def rag():
    print(len(docs))
    resp = llm.invoke(prompt_str)

    print([d.metadata.get("page") for d in docs])
    print(context[:400])