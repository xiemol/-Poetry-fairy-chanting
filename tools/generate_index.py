from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from handerForFiles import *

EMBEDDING_DEVICE = 'cpu'
embeddings = HuggingFaceEmbeddings(model_name='model\m3e-base-huggingface', model_kwargs={'device': EMBEDDING_DEVICE})
if os.path.exists('index\game1_index'):
    vector = FAISS.load_local('index\game1_index', embeddings=embeddings, allow_dangerous_deserialization=True)
# 建立索引，将词向量存入向量数据库
else:
    print("开始")
    docs = []

    all_files = get_all_files('dataSource')
    for file in all_files:
        ext = os.path.splitext(file)[1].lower()  # 获取文件后缀名并转换为小写
        if ext == '.pdf':
            docs.extend(create_document_from_pdf(file))
        elif ext == '.txt':
            docs.extend(create_document_from_txt(file))
        elif ext == '.docx':
            docs.extend(create_document_from_docx(file))
        elif ext == '.html':
            docs.extend(create_document_from_html(file))

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(documents=docs)
    vector = FAISS.from_documents(documents=documents, embedding=embeddings)
    # 保存向量数据库
    vector.save_local('index\game1_index')

retriever = vector.as_retriever()
