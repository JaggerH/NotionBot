import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

import unittest
from utils.pdf_helper import match_title

class TestMatchTitle(unittest.TestCase):
    
    def test_match_title(self):
        test_strings = [
            ('一、经营情况讨论与分析', True),
            ('（一）产品研发', True),
            ('1、电源管理', True),
            ('2、信号链产品', True),
            ('3、功率器件', True),
            ('（二）市场营销', True),
            ('（三）生产运营', True),
            ('四、报告期内核心竞争力分析', True),
            ('1、产品和技术', True),
            ('2、核心团队建设', True),
            ('3、知识产权', True),
            ('4、高新技术企业认定、专精特新企业', True),
            ('(五) 投资状况分析', True),
            ('1. 重大的股权投资', True),
            ('2. 重大的非股权投资', True),
            ('3. 以公允价值计量的金融资产', True),
            ('4. 报告期内重大资产重组整合的具体进展情况', True),
            ('(三)经营计划', True),
            ('无匹配内容', False),
            ('10.未匹配的内容', True)
        ]

        for test_string, expected in test_strings:
            with self.subTest(test_string=test_string):
                match = match_title(test_string)
                if expected:
                    self.assertTrue(match, f"Expected True but got False for '{test_string}'")
                else:
                    self.assertFalse(match, f"Expected False but got True for '{test_string}'")

if __name__ == '__main__':
    unittest.main()