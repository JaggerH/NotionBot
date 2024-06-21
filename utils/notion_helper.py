import os
import pandas as pd

from notion_client import Client
from utils.tushare_helper import stock_basic, daily
# from utils.set_env import set_env

# set_env()
NOTION_API_KEY = os.getenv('NOTION_API_KEY')
NOTION_DATABASE_ID = os.getenv('NOTION_DATABASE_ID')

notion = Client(auth=NOTION_API_KEY)

def create_observe(data):
    """
    创建一个观察记录并将其添加到Notion数据库中。

    参数:
    data (dict): 包含创建观察记录所需数据的字典。
        - "股票代码" (str): 股票的代码。
        - "股票名称" (str): 股票的名称。
        - "推送时间" (str): 推送这条观察记录的时间。
        - "Type" (str): 观察记录的类型，例如“买入”或“卖出”。
        - "届时股价" (float): 观察时的股价。

    返回:
    notion.pages.create的返回值: 创建的Notion页面对象。

    该函数使用Notion API创建一个新的页面，并将其作为观察记录添加到指定的Notion数据库中。
    页面包含股票代码、名称、推送时间、类型和股价等属性。

    使用示例:
    create_observe({
        "股票代码": "000001",
        "股票名称": "平安银行",
        "推送时间": "2024-05-20",
        "Type": "Tracking",
        "届时股价": 100
    })
    """
    return notion.pages.create(
        parent={"database_id": NOTION_DATABASE_ID},
        properties={
            "股票代码": {
                "rich_text": [
                    {
                        "text": {
                            "content": data["股票代码"]
                        }
                    }
                ]
            },
            "股票名称": {
                "rich_text": [
                    {
                        "text": {
                            "content": data["股票名称"]
                        }
                    }
                ]
            },
            "推送时间": {
                "title": [
                    {
                        "text": {
                            "content": data["推送时间"]
                        }
                    }
                ]
            },
            "Type": {
                "select": {
                    "name": data["Type"]
                }
            },
            "推送者": {
                "select": {
                    "name": data["推送者"]
                }
            },
            "届时股价": {
                "number": data["届时股价"]
            }
        }
    )

def extract_stock(text, df):
    """
    从text中抽取存在的股票名称，返回df即从tushare获取的股票基本信息中包含

    Params:
        text: str
        df: pd.DataFrame

    Example:
        df = stock_basic()
        names = extract_stock(text, df)
    """
    stock_names = df['name'].values

    # 根据全称提取
    extracted_names = []
    for name in stock_names:
        if name in text:
            extracted_names.append(name)

    return df[df['name'].isin(extracted_names)]

def build_observe_data(text, date, pusher):
    df = stock_basic()
    df = extract_stock(text, df)
    df['type'] = 'Tracking'
    df['date'] = date
    df['pusher'] = pusher
    price = daily(date)
    df = pd.merge(df, price, on="ts_code", how="left")
    df = df[['symbol', 'name', 'date', 'type', 'close', 'pusher']]
    df = df.rename(columns={
        "symbol": "股票代码",
        "name": "股票名称",
        "date": "推送时间",
        "type": "Type",
        "close": "届时股价",
        "pusher": "推送者"
    })
    return df.to_dict(orient='records')

def array_to_notion_blocks(table, has_column_header=True, has_row_header=True):
    table_width = len(table[0]) if has_column_header else max(len(row) for row in table)

    # Format as Notion blocks
    table_block = {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": table_width,
            "has_column_header": has_column_header,
            "has_row_header": has_row_header,
            "children": []
        }
    }

    for row in table:
        row_block = {
            "object": "block",
            "type": "table_row",
            "table_row": {
                "cells": [[{"type": "text", "text": {"content": cell}}] for cell in row]
            }
        }
        table_block["table"]["children"].append(row_block)

    return table_block

def find_heading_id(page_blocks_response, heading_text):
    """
    查找页面中指定Heading的位置
    :param page_id: 页面ID
    :param heading_text: Heading文本内容
    :return: Heading块的位置
    """
    for i, block in enumerate(page_blocks_response['results']):
        if block['type'] == 'heading_2' and block['heading_2']['rich_text'][0]['text']['content'] == heading_text:
            return block['id']
    return None

def add_blocks_after_heading(page_id, heading_text, blocks):
    """
    在指定Heading后添加块
    :param page_id: 页面ID
    :param heading_text: Heading文本内容
    :param blocks: 要添加的块
    """
    response = notion.blocks.children.list(block_id=page_id)
    heading = find_heading_id(response, heading_text)
    if heading is None:
        print(f"未找到指定的Heading：{heading_text}")
        return
    else:
        notion.blocks.children.append(block_id=page_id, children=blocks, after=heading)

    print(f"已在Heading：{heading_text}后添加内容。")