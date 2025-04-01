import os
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any
import json
from nlp_to_sql import process_natural_language_query

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据库连接配置
DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': '123456',
    'host': 'localhost',
    'port': '5432'
}

def connect_to_db():
    """连接到PostgreSQL数据库"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"数据库连接错误: {e}")
        return None

def execute_query(query: str) -> pd.DataFrame:
    """执行SQL查询并返回DataFrame"""
    conn = connect_to_db()
    if conn is None:
        return None
    
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"查询执行错误:{e}")
        return None
    finally:
        conn.close()

def create_bar_chart(df: pd.DataFrame, x_column: str, y_column: str, title: str = None) -> str:
    """创建柱状图并保存为图片"""
    plt.figure(figsize=(10, 6))
    plt.bar(df[x_column], df[y_column])
    plt.title(title or f'{y_column} vs {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.xticks(rotation=45)
    
    # 保存图片
    image_path = 'chart.png'
    plt.savefig(image_path)
    plt.close()
    return image_path

def process_chart_request(query: str) -> Dict[str, Any]:
    """处理图表请求"""
    # 使用NLP处理用户查询
    nlp_result = process_natural_language_query(query)
    
    if "error" in nlp_result:
        return {"error": "无法解析查询"}
    
    sql_query = nlp_result["sql_query"]
    chart_config = nlp_result["chart_config"]
    
    # 执行SQL查询
    df = execute_query(sql_query)
    if df is None:
        return {"error": "无法获取数据"}
    
    # 创建图表
    chart_path = create_bar_chart(
        df,
        chart_config["x_column"],
        chart_config["y_column"],
        chart_config["title"]
    )
    
    return {
        "success": True,
        "chart_path": chart_path,
        "data": df.to_dict(orient='records'),
        "sql_query": sql_query
    }

# 使用示例
if __name__ == "__main__":
    # 测试查询
    # result = process_chart_request("显示各个类别的数量统计")
    # result = process_chart_request("显示各个类产品的订单数量")
    # print(json.dumps(result, ensure_ascii=False, indent=2))
    result = process_chart_request("显示各类产品订单总金额")
    print(json.dumps(result, ensure_ascii=False, indent=2))
