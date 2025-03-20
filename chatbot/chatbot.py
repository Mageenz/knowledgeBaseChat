import os
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_community.chat_models import ChatOllama

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# 使用本地部署的 mistral 模型，启用流式输出
llm = ChatOllama(
    model="mistral",
    temperature=0.7,
    callbacks=[StreamingStdOutCallbackHandler()]
)

def chatbot(state: State):
    # 将消息转换为正确的格式
    messages = [
        SystemMessage(content="你是一个有帮助的AI助手。请用中文回答用户的问题，回答要详细、准确、有礼貌。")
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
    return {"messages": [{"role": "assistant", "content": response.content}]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# 运行聊天机器人
messages = []
while True:
    user_input = input("\n你: ")
    if user_input.lower() in ['退出', 'quit', 'exit']:
        break
    messages.append({"role": "user", "content": user_input})
    print("助手: ", end="", flush=True)  # 打印助手前缀，不换行
    response = graph.invoke({"messages": messages})
    assistant_message = response["messages"][-1]
    messages.append(assistant_message)
    print()  # 最后换行