from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from chatmodel import openai
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_community.document_loaders import WebBaseLoader
#加载网页
loader = WebBaseLoader(
    web_paths=["https://www.gushiwen.cn/shiwens/default.aspx?page=1&tstr=&astr=&cstr=%E5%94%90%E4%BB%A3&xstr=",
               "https://www.gushiwen.cn/shiwens/default.aspx?page=2&tstr=&astr=&cstr=%E5%94%90%E4%BB%A3&xstr=",
]
)
#文档中的词转为词向量
docs=loader.load()

from langchain_huggingface import HuggingFaceEmbeddings
EMBEDDING_DEVICE="cpu"
embeddings = HuggingFaceEmbeddings(model_name="model\m3e-base-huggingface",
                                       model_kwargs={"device": EMBEDDING_DEVICE})
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
#生成 词切分器
text_splitter = RecursiveCharacterTextSplitter()
#对load进的文档进行分词
documents = text_splitter.split_documents(documents=docs)
#建立索引：将词向量存入向量数据库
vector = FAISS.from_documents(documents=documents, embedding=embeddings)


def game3():
    chat_model = openai()

    retriever = vector.as_retriever()

    # 生成ChatModel会话提示词
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "接下来我将给你几个关键词，请你依据这些关键词帮我生成一首诗。不要有多余信息")
    ])
    # 生成含有历史信息的检索链
    retriever_chain = create_history_aware_retriever(chat_model, retriever, prompt)
    # 继续对话，记住检索到的文档等信息
    prompt = ChatPromptTemplate.from_messages([
        ("system", "接下来我将给你几个关键词，请你依据这些关键词帮我生成一首诗，这首诗至少四句话。你的上下文是优秀的唐诗代表，请从{context}学习如何写出好的诗"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    # 生成回答的文档链，主要是用prompt告诉chat_model怎么生成回答
    document_chain = create_stuff_documents_chain(chat_model, prompt)

    # 检索链＋文档链形成的整合链
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    return retrieval_chain
    #return chain
