import os
import tushare as ts

pro = ts.pro_api(os.getenv('TUSHARE_APIKEY'))

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