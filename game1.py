import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from chatmodel import openai
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


#RAG生成书籍介绍
def game1():
    chat_model = openai()
    EMBEDDING_DEVICE = "cpu"
    embeddings = HuggingFaceEmbeddings(model_name="model\m3e-base-huggingface",
                                       model_kwargs={"device": EMBEDDING_DEVICE})

    vector = FAISS.load_local('index\game1_index', embeddings=embeddings, allow_dangerous_deserialization=True)
    # 建立索引：将词向量存入向量数据库
    retriever = vector.as_retriever()

    # 生成ChatModel会话提示词
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "根据以上对话历史，生成一个检索查询，以便查找对话相关信息")
    ])
    # 生成含有历史信息的检索链
    retriever_chain = create_history_aware_retriever(chat_model, retriever, prompt)
    # 继续对话，记住检索到的文档等信息
    prompt = ChatPromptTemplate.from_messages([
        ("system", "根据上下文等信息来回答用户的问题{context}，请注意你要以一个概括的角度说书里讲了什么，可以摘用对应书中的原话，但是不要出现超脱了古代范围的词"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    # 生成回答的文档链，主要是用prompt告诉chat_model怎么生成回答
    document_chain = create_stuff_documents_chain(chat_model, prompt)

    # 检索链＋文档链形成的整合链
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    return retrieval_chain

