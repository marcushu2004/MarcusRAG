# This part and src package still developing. DO NOT USE THIS. PLEASE USE loaders.py
# 这部分和src包仍在开发。请不要使用。请使用loaders.py。

from langchain_community.chat_models import ChatOllama
from src.indexing import format_context
from src.pdf_loader import pdf_loader,all_docs
from src.rag import rag

pdf_loader()
# This is the Question that you want to ask. You can change it.
# 这部分是你想要问的问题。你可以修改它。
question_main = "How is the policy network trained in AlphaGo?"

# This is the model, you can change it that you want. Now is Alibaba Qianwen model.
# 这部分是传入的模型，你可以根据需要来修改。目前采用阿里千问模型。
llm_main = ChatOllama(model="qwen2.5:3b")

context_main = format_context(all_docs)
# This part is the description that you want to give the AI. You can change the style that you want.
# 这部分是给AI的描述。你可以根据你喜欢的风格来修改。
prompt_str_main = f"You are a assistant. Answer the questions ONLY using the context. If you can not find the answer. Use citations (page numbers) in Sources.,{context_main},{question_main}"

rag()