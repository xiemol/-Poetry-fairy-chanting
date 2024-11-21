import os
os.environ["ZHIPUAI_API_KEY"] = "19ffa5597bd867ae38b9f2a356551e69.EjQJJKQAhVrGomtU"

from langchain_community.chat_models import ChatZhipuAI


#环境变量后期在将osvariable.py中相关密钥替换成自己的即可
def openai():
    zhipuai_chat_model = ChatZhipuAI(api_ley="19ffa5597bd867ae38b9f2a356551e69.EjQJJKQAhVrGomtU", model="glm-4-flash")
    return zhipuai_chat_model