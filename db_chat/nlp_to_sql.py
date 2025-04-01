from typing import Annotated, Dict, Any, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
import json
from langchain_community.chat_models import ChatOllama
from IPython.display import Image, display


class TableSchema:
    def __init__(self, table_name: str, columns: List[Dict[str, str]]):
        self.table_name = table_name
        self.columns = columns

    def to_description(self) -> str:
        """将表结构转换为描述性文本"""
        column_descriptions = []
        for col in self.columns:
            desc = f"- {col['name']} ({col['type']})"
            if col.get('description'):
                desc += f": {col['description']}"
            column_descriptions.append(desc)
        
        return f"表名: {self.table_name}\n列信息:\n" + "\n".join(column_descriptions)

# 系统提示词模板
SYSTEM_PROMPT_TEMPLATE = """你是一个专业的SQL专家。你的任务是将用户的自然语言问题转换为PostgreSQL查询语句。

数据库表结构信息如下：
{table_schemas}

请严格遵循以下规则：
1. 你的回答必须且只能包含一个SQL查询语句
2. 不要包含任何解释、注释或其他文本
3. 确保SQL语句符合PostgreSQL语法
4. 使用提供的表名和列名
5. 确保查询安全，避免SQL注入风险
6. 使用合适的表连接和聚合函数
7. 添加适当的WHERE条件过滤数据
8. 使用ORDER BY进行排序
9. 使用LIMIT限制结果数量
10. 不要在SQL语句前后添加任何其他内容
"""

CHART_CONFIG_PROMPT = """你是一个数据可视化专家。请根据以下SQL查询生成合适的图表配置。

SQL查询：
{sql_query}

请生成一个JSON格式的图表配置，包含以下字段：
1. type: 图表类型（可选值：bar, line, pie, table）
2. x_column: x轴列名（如果是表格类型，则不需要）
3. y_column: y轴列名（如果是表格类型，则不需要）
4. title: 图表标题

注意：
1. 确保列名与SQL查询结果中的列名完全匹配
2. 对于聚合查询（如COUNT、AVG等），使用查询中定义的别名
3. 对于普通查询，使用表格类型展示
4. 返回的必须是有效的JSON格式
5. 不要在JSON前后添加任何其他内容

示例：
对于查询：SELECT category, COUNT(*) as category_count FROM products GROUP BY category
应该生成如下配置：
{{
    "type": "bar",
    "x_column": "category",
    "y_column": "category_count",
    "title": "产品类别数量统计"
}}
"""

class State(TypedDict):
    messages: Annotated[list, add_messages]
    sql_query: str
    chart_config: Dict[str, Any]

def create_nlp_to_sql_graph():
    table_schemas = [
        TableSchema(
            table_name="products",
            columns=[
                {"name": "id", "type": "INTEGER", "description": "产品ID"},
                {"name": "name", "type": "VARCHAR(255)", "description": "产品名称"},
                {"name": "category", "type": "VARCHAR(100)", "description": "产品类别"},
                {"name": "price", "type": "DECIMAL(10,2)", "description": "产品价格"},
                {"name": "stock", "type": "INTEGER", "description": "库存数量"},
                {"name": "created_at", "type": "TIMESTAMP", "description": "创建时间"}
            ]
        ),
        TableSchema(
            table_name="orders",
            columns=[
                {"name": "id", "type": "INTEGER", "description": "订单ID"},
                {"name": "product_id", "type": "INTEGER", "description": "产品ID"},
                {"name": "quantity", "type": "INTEGER", "description": "购买数量"},
                {"name": "total_amount", "type": "DECIMAL(10,2)", "description": "订单总金额"},
                {"name": "order_date", "type": "TIMESTAMP", "description": "订单日期"},
                {"name": "customer_id", "type": "INTEGER", "description": "客户ID"}
            ]
        ),
        TableSchema(
            table_name="customers",
            columns=[
                {"name": "id", "type": "INTEGER", "description": "客户ID"},
                {"name": "name", "type": "VARCHAR(255)", "description": "客户名称"},
                {"name": "email", "type": "VARCHAR(255)", "description": "客户邮箱"},
                {"name": "created_at", "type": "TIMESTAMP", "description": "注册时间"}
            ]
        )
    ]
    # 创建状态图
    graph_builder = StateGraph(State)
    
    # 初始化LLM
    llm = ChatOpenAI(
        model="qwen2.5:14b",
        temperature=0.7,
        openai_api_base="http://192.168.0.118:11434/v1",
        openai_api_key="sk-e0e10153553243be94e8666fe0deca85"
    )
    # 使用本地部署的 mistral 模型
    # llm = ChatOllama(
    #     model="mistral",
    #     temperature=0.7
    # )

    # 生成包含表结构的系统提示词
    table_schema_text = "\n\n".join(schema.to_description() for schema in table_schemas)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(table_schemas=table_schema_text)
   
    def generate_sql(state: State) -> State:
        """生成SQL查询"""
        messages = state["messages"]
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [SystemMessage(content=system_prompt)] + messages
        
        response = llm.invoke(messages)
        state["sql_query"] = response.content
        return state
    
    def generate_chart_config(state: State) -> State:
        """生成图表配置"""
        sql_query = state["sql_query"]
        # print(f"sql_query:{sql_query}")
        
        chart_prompt = CHART_CONFIG_PROMPT.format(sql_query=sql_query)
        # print(f"chart_prompt:{chart_prompt}")
        
        messages = [SystemMessage(content=chart_prompt)]
        response = llm.invoke(messages)

        # print(f"response:{response}")
        
        try:
            # 尝试解析JSON响应
            chart_config = json.loads(response.content)
            # print(f"chart_config:{chart_config}")
            state["chart_config"] = chart_config
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing chart config: {e}")
            # 如果解析失败，使用默认的表格配置
            state["chart_config"] = {
                "type": "table",
                "columns": ["*"],
                "title": "查询结果"
            }
        
        return state
    
    def should_end(state: State) -> str:
        """决定是否结束处理"""
        if state.get("sql_query"):
            return "end"
        return "continue"
    
    # 添加节点
    graph_builder.add_node("generate_sql", generate_sql)
    graph_builder.add_node("generate_chart", generate_chart_config)
    
    # 添加边
    graph_builder.add_edge(START, "generate_sql")
    graph_builder.add_edge("generate_sql", "generate_chart")
    graph_builder.add_conditional_edges(
        "generate_chart",
        should_end,
        {
            "continue": "generate_sql",
            "end": END
        }
    )
    try:
        display(Image(graph_builder.get_graph().draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        pass
    return graph_builder.compile()

def process_natural_language_query(query: str) -> Dict[str, Any]:
    """处理自然语言查询"""
    graph = create_nlp_to_sql_graph()
    
    # 初始化状态
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "sql_query": "",
        "chart_config": {}
    }
    
    # 运行图
    final_state = graph.invoke(initial_state)
    
    return {
        "sql_query": final_state["sql_query"],
        "chart_config": final_state["chart_config"]
    }