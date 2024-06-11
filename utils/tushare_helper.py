import os
import tushare as ts
print(os.getenv('TUSHARE_API_KEY'))
pro = ts.pro_api(os.getenv('TUSHARE_API_KEY'))

def stock_basic():
    return pro.stock_basic(**{
        "ts_code": "",
        "name": "",
        "exchange": "",
        "market": "",
        "is_hs": "",
        "list_status": ""
    }, fields=[ "ts_code", "symbol", "name", "area", "industry", "cnspell", "market", "list_date", "act_name", "act_ent_type" ])

def daily(date):
    return pro.daily(**{
        "ts_code": "",
        "trade_date": date.replace("-", ""),
        "start_date": "",
        "end_date": "",
        "offset": "",
        "limit": ""
    }, fields=[ "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount" ])

def find_similar_rows(df, target):
    """
    在 symbol 或 name 列中查找包含 target 字符串的行
    
    :param df: pandas DataFrame
    :param target: 目标字符串
    :return: 包含目标字符串的行的 DataFrame

    target = '万科'
    result_df = find_similar_rows(df, target)
    """
    mask = df['symbol'].str.contains(target, case=False, na=False) | df['name'].str.contains(target, case=False, na=False)
    return df[mask]