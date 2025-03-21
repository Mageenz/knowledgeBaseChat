import os
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOllama

from vectorstore_manager import VectorStoreManager

class State(TypedDict):
    messages: Annotated[list, add_messages]

# 初始化向量数据库管理器
vectorstore_manager = VectorStoreManager()

graph_builder = StateGraph(State)

# 使用本地部署的 mistral 模型
llm = ChatOllama(
    model="mistral",
    temperature=0.7
)

def chatbot(state: State):
    # 获取最后一条用户消息
    last_message = state["messages"][-1]
    if isinstance(last_message, dict):
        last_message = last_message["content"]
    else:
        last_message = last_message.content
    
    # 从知识库中检索相关文档
    docs = vectorstore_manager.search(last_message, k=3)
    
    # 构建系统提示，包含检索到的知识
    system_prompt = "你是一个有帮助的AI助手。请用中文回答用户的问题，回答要详细、准确、有礼貌。\n\n"
    system_prompt += "以下是相关的知识：\n"
    for doc in docs:
        system_prompt += f"- 内容：{doc['content']}\n"
        if doc.get('metadata'):
            system_prompt += f"  来源：{doc['metadata'].get('source', '未知')}\n"
            if 'page' in doc['metadata']:
                system_prompt += f"  页码：{doc['metadata']['page']}\n"
        system_prompt += "\n"
    
    # 将消息转换为正确的格式
    messages = [
        SystemMessage(content=system_prompt)
    ]
    
    for msg in state["messages"]:
        if isinstance(msg, dict):
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        else:
            messages.append(msg)
    
    response = llm.invoke(messages)
    
    # 确保返回的消息格式正确
    return {
        "messages": [
            {"role": "user", "content": last_message},
            {"role": "assistant", "content": response.content}
        ]
    }

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# 运行聊天机器人
# messages = []
# while True:
#     user_input = input("\n你: ")
#     if user_input.lower() in ['退出', 'quit', 'exit']:
#         break
#     messages.append({"role": "user", "content": user_input})
#     print("助手: ", end="", flush=True)  # 打印助手前缀，不换行
#     response = graph.invoke({"messages": messages})
#     assistant_message = response["messages"][-1]
#     messages.append(assistant_message)
#     print()  # 最后换行