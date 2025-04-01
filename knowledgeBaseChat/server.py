from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from vectorstore_manager import VectorStoreManager
import asyncio
import time
import logging
import os
from datetime import datetime
from langchain_community.chat_models import ChatOllama

# 使用本地部署的 mistral 模型
llm = ChatOllama(
    model="mistral",
    temperature=0.7
)

# 配置日志
def setup_logger():
    # 创建logs目录
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 生成日志文件名，包含日期
    log_filename = f'logs/chat_{datetime.now().strftime("%Y%m%d")}.log'
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    return logging.getLogger(__name__)

# 初始化日志
logger = setup_logger()

# 初始化向量数据库管理器
vectorstore_manager = VectorStoreManager()

app = FastAPI(
    title="知识库聊天API",
    description="基于向量数据库的智能问答系统API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境建议设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

class ChatResponse(BaseModel):
    messages: List[Dict[str, str]]

def stream_chat(messages: List[Dict[str, str]]):
    """同步流式聊天处理函数"""
    try:
        # 获取最后一条用户消息
        last_message = messages[-1]["content"]
        logger.info(f"收到用户消息: {last_message}")
        
        # 从知识库中检索相关文档
        docs = vectorstore_manager.search(last_message, k=3)
        logger.info(f"检索到 {len(docs)} 条相关文档")
        
        # 构建系统提示，包含检索到的知识
        system_prompt = "你是一个有帮助的AI助手。请用中文回答用户的问题，回答要详细、准确、有礼貌。\n\n"
        
        if not docs:
            system_prompt += "抱歉，我在知识库中没有找到与您问题相关的信息。我会尽力根据我的知识来回答您的问题。\n\n"
        else:
            system_prompt += "以下是相关的知识：\n"
            for doc in docs:
                system_prompt += f"- 内容：{doc['content']}\n"
                if doc.get('metadata'):
                    system_prompt += f"  来源：{doc['metadata'].get('source', '未知')}\n"
                    if 'page' in doc['metadata']:
                        system_prompt += f"  页码：{doc['metadata']['page']}\n"
                system_prompt += "\n"
        
        # 将消息转换为正确的格式
        chain_messages = [
            SystemMessage(content=system_prompt)
        ]
        
        for msg in messages:
            if msg["role"] == "user":
                chain_messages.append(HumanMessage(content=msg["content"]))
            else:
                chain_messages.append(AIMessage(content=msg["content"]))
        
        logger.info("开始生成回复")
        # 同步流式输出
        for chunk in llm.stream(chain_messages):
            if chunk.content:
                yield f"data: {json.dumps({'content': chunk.content})}\n\n"
        
        logger.info("回复生成完成")
                
    except Exception as e:
        error_msg = f"错误：{str(e)}"
        logger.error(error_msg, exc_info=True)
        yield f"data: {json.dumps({'content': error_msg})}\n\n"

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    聊天接口（流式输出）
    
    Args:
        request: 包含消息历史的请求体
        
    Returns:
        流式响应
    """
    logger.info("收到新的聊天请求")
    return StreamingResponse(
        stream_chat(request.messages),
        media_type="text/event-stream"
    )

@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    logger.info("收到健康检查请求")
    return {"status": "healthy"}

if __name__ == "__main__":
    logger.info("启动API服务")
    uvicorn.run(app, host="0.0.0.0", port=8000) 