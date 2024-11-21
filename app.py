from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from typing import List
from dailychat import chat
from game1 import game1
from game2 import game2, extract_first_number_advanced
from game3 import game3
import logging
import osvariables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
chat_chain1 = chat('yahuan')
chat_chain2 = chat('dashi')
book_chain=game1()
charge_chain=game2()
poem_chain=game3()
print("step2:模型初始化完毕")


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    input: str
    history: List[ChatMessage]


class ChatResponse(BaseModel):
    answer: str

class BookRequest(BaseModel):
    input: str

class ChargeRequest(BaseModel):
    input: str

@app.post("/chat1")
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received request: {request}")
    # 将历史对话转换为适合您的chat函数的格式
    history = [{"role": msg.role, "content": msg.content} for msg in request.history]

    # 调用chat函数，传入当前输入和历史对话
    answer = chat_chain1.invoke(
        input={
            "input": request.input,
            "history": history
        }
    )
    print(answer)
    return {"answer":answer['response']}

@app.post("/chat2")
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received request: {request}")
    # 将历史对话转换为适合您的chat函数的格式
    history = [{"role": msg.role, "content": msg.content} for msg in request.history]

    # 调用chat函数，传入当前输入和历史对话
    answer = chat_chain2.invoke(
        input={
            "input": request.input,
            "history": history
        }
    )
    print(answer)
    return {"answer":answer['response']}

@app.post("/game1")
def chat(request: BookRequest):
    input_text = f"请你以古人的视角为我讲一下《" + request.input + "》这本书，不要出现现代词汇,请注意保留章节主要信息，字数不要超过150字。"
    answer = book_chain.invoke({
        "input": input_text,
        "chat_history": []
    })
    return {"answer":answer["answer"]}

@app.post("/game2")
def chat(request: ChargeRequest):
    input_text = request.input
    answer = charge_chain.invoke(input_text)
    score = extract_first_number_advanced(answer['text'])
    return {"answer":answer["text"],"score":score}

@app.post("/game3")
def chat(request: BookRequest):
    input_text = request.input
    answer = poem_chain.invoke({
        "input": input_text,
        "chat_history": ["user","请只生成一首诗，至少四句话。请注意，七言绝句指的是有每一句话有七个字，一共有四句话，不能少也不能多。利用我给你的关键字。不要有多余信息。"]
    })
    return {"answer":answer["answer"]}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=app, host="localhost", port=8001)