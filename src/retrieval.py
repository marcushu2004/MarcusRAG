from main import context_main, prompt_str_main, question_main
from src.indexing import vectorstore, format_context

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":4, "fetch_k":20})

# This is the Question that you want to ask. You can change it.
# 这部分是你想要问的问题。你可以修改它。
question = question_main# "How is the policy network trained in AlphaGo?"

docs = retriever.invoke(question)
context = context_main # format_context(docs)

# This part is the description that you want to give the AI. You can change the style that you want.
# 这部分是给AI的描述。你可以根据你喜欢的风格来修改。
prompt_str = prompt_str_main