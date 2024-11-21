import os
from typing import Dict, Any
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from chatmodel import openai
#后期再在这里加上别的人物，模型换成国内模型
#丫鬟
person_template1 = """你是一款游戏中，中国唐朝里扬州的一位青楼女子。你的任务是和一名叫李怀特的游戏角色对话，
请注意游戏中的李怀特就是现实世界的大诗人李白，如果涉及到李白的信息，请统一换成李怀特。
请对玩家称呼为‘官人’

    当前对话历史：
    {history}

    玩家: {input}
    你: """

#大师
person_template2 = """你是一款游戏中，中国唐朝的一位隐居的大师。你的任务是和一名叫李怀特的游戏角色对话，
你很聪明，了解很多内容，对道学，《道德经》，《论语》等经典著作都有深厚的了解。

    当前对话历史：
    {history}

    玩家: {input}
    你: """

all_person={
    "yahuan":person_template1,
    "dashi":person_template2,
}

def chat(character):#传入角色名
    # 修改人设提示模板，包含历史对话
    person_template=all_person[character]
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=person_template
    )

    chat_model = openai()
    conv_chain: ConversationChain = ConversationChain(
        llm=chat_model,
        memory=ConversationBufferMemory(memory_key="history"),
        prompt=prompt,
    )
    return conv_chain


