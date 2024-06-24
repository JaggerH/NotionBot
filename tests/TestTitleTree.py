import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
sys.path.append('..')

import unittest
from utils.pdf_helper import match_title, extract_title_numbers, Tree

class TestMatchTitle(unittest.TestCase):
    
    def test_title_tree(self):
        lines = [
            ('2.1 2023年核心技术创新', 1),
            ('2.1.1 坚持长期投入，掌控底层核心技术', 2),
            ('1、芯片领域', 3),
            ('2、数据库领域', 3),
            ('3、操作系统领域', 3),
            ('2.1.2 技术创新引领，持续提升产品竞争力', 2),
            ('1、高速网络', 3),
            ('2、算力基础设施', 3),
            ('3、数字能源', 3),
            ('2.2 2023年经营情况回顾', 1),
            ('2.2.1 行业发展情况', 2),
            ('1、国内市场', 3),
            ('2、国际市场', 3),
            ('2.2.2 本集团业务和财务分析', 2),
        ]

        tree = Tree('test_title_tree')
        for title, depth in lines:
            if match_title(title):
                node = tree.add_node(title)
                self.assertEqual(node.depth, depth)

if __name__ == '__main__':
    unittest.main()