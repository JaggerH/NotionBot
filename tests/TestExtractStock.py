import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

import unittest
import pandas as pd
from utils.notion_helper import extract_stock
from utils.tushare_helper import stock_basic

class TestExtractStock(unittest.TestCase):

    def setUp(self):
        # 在每个测试之前运行，可以用来设置测试环境
        self.df = stock_basic()

    def test_extract_stock_single(self):
        text = "今天我关注了平安银行的股票走势。"
        expected_result = self.df[self.df['name'] == '平安银行']
        result = extract_stock(text, self.df)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_extract_stock_multiple(self):
        text = "今天我关注了平安银行和贵州茅台的股票走势。"
        expected_result = self.df[self.df['name'].isin(['平安银行', '贵州茅台'])]
        result = extract_stock(text, self.df)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_extract_stock_none(self):
        text = "今天我没有关注任何股票。"
        expected_result = self.df[self.df['name'].isin([])]
        result = extract_stock(text, self.df)
        pd.testing.assert_frame_equal(result, expected_result)

if __name__ == '__main__':
    unittest.main()
