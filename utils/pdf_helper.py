import re
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import pdfplumber
from typing import Dict
from paddleocr import PPStructure, paddleocr, PaddleOCR, draw_structure_result
import logging
import cv2
import math
from PIL import Image

paddleocr.logging.disable(logging.DEBUG)
logger = logging.getLogger('pdf_helper')

class LazyPaddleEngine:
    def __init__(self):
        self._ocr_engine = None

    @property
    def ocr_engine(self):
        if self._ocr_engine is None:
            self._ocr_engine = PPStructure(table=False, ocr=True)
        return self._ocr_engine
    
lazy_paddle_engine = LazyPaddleEngine()
# ocr_engine = lazy_paddle_engine.ocr_engine

title_patterns = {
    'chinese_brackets': r"（[一二三四五六七八九十]+）",
    'arabic_brackets': r"\([一二三四五六七八九十]+\)",
    'chinese_number_comma': r"[一二三四五六七八九十]+、",
    'arabic_number_comma': r"[1-9]\d*、",
    'arabic_number_dot': r"[1-9]\d*\.\s+",
    # 'arabic_dot_level': r"\d+(\.\d+)+\s+", # 2.1.1
    'arabic_dot_level': r"\d+(\.\d+)+(?=\s(?!.*\b(亿元|元|名|%)))", # 2.1.1
}

#=========================================
#              通用方法
# 以下内容都是对PDF的处理具有通用意义的
#=========================================
def find_pages_with_keyword(pdf_path, keyword, max_pages=20):
    # 运行效率比pdfplumber.page.search快很多倍
    document = fitz.open(pdf_path)
    pages_with_keyword = []
    for page_num in range(min(len(document), max_pages)):
        page = document.load_page(page_num)
        text = page.get_text()
        text = text.replace(" ", "")
        if keyword in text:
            text_lines = text.split('\n')
            for line in text_lines:
                if line == keyword:
                    pages_with_keyword.append(page_num + 1)  # 页码从1开始

    pages_with_keyword = list(set(pages_with_keyword))  # 去重
    return pages_with_keyword

def extract_text_with_percent_bbox(page, bbox=(0, 0, 1, 1)):
    if not isinstance(page, pdfplumber.page.Page):
        raise Exception("extract_text_with_percent_bbox requires a pdfplumber.page.Page object")

    width, height = page.width, page.height
    left, top, right, bottom = bbox
    # 定义要裁剪的区域
    cropping_box = (
        width * left,  # 左边界
        height * top,  # 上边界
        width * right,  # 右边界
        height * bottom  # 下边界
    )

    cropped_page = page.within_bbox(cropping_box)
    text = cropped_page.extract_text()
    
    return text

def extract_text_with_percent_bbox_from_pdf(pdf_path, page_number, bbox=(0, 0, 1, 1)):
    """
    从指定页面提取文本内容，并尝试过滤页头和页尾。

    参数:
        pdf_path (str): PDF 文件路径。
        page_number (int): 页码（从1开始）。
        bbox (tuple): 按页面比例框选bbox。

    返回:
        str: 过滤后的页面文本内容。
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        return extract_text_with_percent_bbox(page, bbox)

def crop_page(page, scale_factor=2):
    """
        只支持fitz.Page对象
    """
    if not isinstance(page, fitz.Page):
        raise Exception("crop_page requires a fitz.Page object")
        
    mat = fitz.Matrix(scale_factor, scale_factor)
    pm = page.get_pixmap(matrix=mat, alpha=False)
    # if width or height > 2000 pixels, don't enlarge the image
    # if pm.width > 2000 or pm.height > 2000:
    #     pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
    
    img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
    return img

def crop_page_from_pdf(pdf_path, page_number):
    document = fitz.open(pdf_path)
    page = document.load_page(page_number - 1)
    # pix = page.get_pixmap()
    # image_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    # if pix.n == 4:
    #     image_data = cv2.cvtColor(image_data, cv2.COLOR_BGRA2BGR)
    # pix.save('temp.png')
    # return image_data
    return crop_page(page)

def is_page_one_column_or_two_column(image):
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')
    result = ocr.ocr(image, cls=True)
    boxes = [line[0] for line in result[0]]
    column_counts = {'left': 0, 'right': 0}
    for box in boxes:
        center_x = (box[0][0] + box[2][0]) / 2
        if center_x < image.shape[1] / 2:
            column_counts['left'] += 1
        else:
            column_counts['right'] += 1
    if column_counts['left'] > 0 and column_counts['right'] > 0:
        left_to_right_ratio = column_counts['left'] / column_counts['right']
        if 0.5 <= left_to_right_ratio <= 2:
            return 2
    return 1

def get_pdf_table_of_content(filepath: str) -> Dict[int, str]:
    """
    读取 PDF 自带的目录信息
    """
    bookmarks = []
    with fitz.open(filepath) as doc:
        toc = doc.get_toc()  # [[lvl, title, page, …], …]
        for level, title, page in toc:
            bookmarks.append({'chapter': title, 'page': page, 'level': level})
    return bookmarks

def extract_text(page, bbox):
    if isinstance(page, fitz.Page):
        pdf = pdfplumber.open(page.parent.name)
        page = pdf.pages[page.number]
    cropped = page.within_bbox(bbox)
    text = cropped.extract_text()
    return text

def extract_table(page, bbox):
    if isinstance(page, fitz.Page):
        pdf = pdfplumber.open(page.parent.name)
        page = pdf.pages[page.number]
    cropped = page.within_bbox(bbox)
    tables = cropped.extract_table()
    return tables

def scale_bbox(bbox, scale_factor):
    return [coord / scale_factor for coord in bbox]
#=========================================
#              提取章节内容
#=========================================
def extract_chapters_from_page(text, pdf_path, pages):
    """
    从文本中提取章节名称和起始页码。
    参数:
        text (str): 包含章节信息的文本。
        pdf_path (str): PDF 文件路径。
        pages (list): 页码列表。
        pdf_path和pages都是为了异形文档准备的
    返回:
        list of dict: 提取的章节信息，每个章节信息包含"chapter"和"start"。
    """
    if isinstance(text, list):
        text = "\n".join(text)
        
    chapters = []
    
    pattern_dots = re.compile(r'^(\S.*?)\s*[·\.]+\s*(\d+)$', re.MULTILINE) # 第四节 公司治理....... 72
    pattern_page_at_start = re.compile(r'^\s*(\d+)\s+(\S.*)$', re.MULTILINE) # 37 管理层讨论与分析
    pattern_dot_validate = re.compile(r'[·\.]{6}')
    
    # 如果匹配到连续6个点的模式
    if pattern_dot_validate.search(text):
        logger.info("匹配到文本为模式一,形如\"第四节 公司治理....... 72\",采用pattern_dots模式")
        matches_dots = pattern_dots.findall(text)
        # for chapter, page_num in matches_dots:
        #     page_num = int(page_num)
        #     if chapter.strip():
        #         chapters.append({"chapter": chapter.strip(), "page": page_num})
        chapters.extend({"chapter": chapter.strip(), "page": int(page_num)} for chapter, page_num in matches_dots if chapter.strip())
    # 如果匹配到页码在开头的模式
    else:
        logger.info(f'{pdf_path} 异形文档')
        if len(pages) != 1: 
            raise ValueError("异形文档必须只有一页")
        
        image = crop_page_from_pdf(pdf_path, pages[0]) # 其实目前已经做到pages范围缩小到只有一页了
        page_layout = recongize_layout(image)
        column = is_page_one_column_or_two_column(page_layout)
        if column == 2:
            logger.info(f'{pdf_path}文档左右布局')
            left_text = extract_text_with_percent_bbox_from_pdf(pdf_path, pages[0], (0, 0.1, 0.5, 0.9)) 
            right_text = extract_text_with_percent_bbox_from_pdf(pdf_path, pages[0], (0.5, 0.1, 1, 0.9))
            text = left_text + right_text
            
        matches_start = pattern_page_at_start.findall(text)
        chapters.extend({"chapter": chapter.strip(), "page": int(page_num)} for page_num, chapter in matches_start if chapter.strip())
        # for page_num, chapter in matches_start:
        #     page_num = int(page_num)
        #     if chapter.strip():
        #         chapters.append({"chapter": chapter.strip(), "page": page_num})
    
    return chapters
    
def extract_chapters(pdf_path):
    """
    提取 PDF 中的章节信息。
    https://www.notion.so/d75758fbb9f0473f87f47febd40dd9dd?v=b7715a608c494e35a5663e203701f3f3&p=6ccc837f1b424a6499ef434c6cd109a1&pm=s

    参数:
        pdf_path (str): PDF 文件路径。
        keyword (str): 用于查找目录的关键字。

    返回:
        list of dict: 提取的章节信息，每个章节信息包含"chapter", "start", "end"。
                      如果未找到包含关键字的页面或未找到包含连续6个点的内容，则返回相应的错误信息。
    """
    chapters = get_pdf_table_of_content(pdf_path)
    if len(chapters) > 3: 
        logger.info(f"{pdf_path}自带标签符合条件")
        chapters = [item for item in chapters if item['level'] == 1] # 仅保留level==1的item
        return chapters

    # ----------- 筛选存在"目录"的页面 -----------
    keyword = '目录'
    pages = find_pages_with_keyword(pdf_path, keyword)
    if not pages: 
        logger.error(f"{pdf_path}未找到包含关键词'{keyword}'的页面。")
        return []
    logger.info(f"find_pages_with_keyword: {pdf_path}, pages: {pages}")
    # ----------- 筛选存在"目录"的页面 -----------
    
    # 缩小页面范围
    pages_content = [extract_text_with_percent_bbox_from_pdf(pdf_path, page, (0, 0.1, 1, 0.9)) for page in pages]
    
    chapters = extract_chapters_from_page(pages_content, pdf_path, pages)
    return chapters

#=========================================
#              PPOCR相关的代码
#=========================================
def recongize_layout(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    page_layout = lazy_paddle_engine.ocr_engine(img)
    for line in page_layout: line.pop('img')
    return page_layout

def debug_layout(image, page_layout):
    font_path = 'simfang.ttf' # PaddleOCR下提供字体包
    im_show = draw_structure_result(image, page_layout, font_path=font_path)
    im_show = Image.fromarray(im_show)
    im_show.show()

#=========================================
#       根据page_layout提取页面内容
#=========================================
def bbox_correct(table_layout, page_bbox):
    """
    这个函数存在时因为有时候pdfplumber会抽风，生成的table bbox区间会超出页面
    """
    bbox = table_layout['bbox']
    bbox[0] = bbox[0] if bbox[0] >= 0 else 0
    bbox[1] = bbox[1] if bbox[1] >= 0 else 0
    bbox[2] = bbox[2] if bbox[2] <= page_bbox[2] else page_bbox[2]
    bbox[3] = bbox[3] if bbox[3] <= page_bbox[3] else page_bbox[3]
    table_layout['bbox'] = bbox
    return table_layout
    
def get_header_bottom_border(page_layout):
    headers = [block for block in page_layout if block['type'] == 'header']
    bottom_borders = [header['bbox'][3] for header in headers]
    return max(bottom_borders) if bottom_borders else 0

def get_footer_top_border(page_layout, page_height):
    footers = [block for block in page_layout if block['type'] == 'footer']
    footer_borders = [footer['bbox'][3] for footer in footers]
    return max(footer_borders) if footer_borders else page_height

def split_page_layout_by_table(blocks, table, width, height):
    _blocks = []
    table_bbox = table['bbox']
    table_top = table_bbox[1]
    table_bottom = table_bbox[3]
    
    for block in blocks:
        block_top, block_bottom = block['bbox'][1], block['bbox'][3]
        
        if table_top > block_top and table_bottom < block_bottom:
            # Table is within the block, split into two and add the table
            _blocks.append({'type': 'text', 'bbox': [0, block_top, width, table_top]})
            _blocks.append(table)
            _blocks.append({'type': 'text', 'bbox': [0, table_bottom, width, block_bottom]})
        elif table_top <= block_top and table_bottom >= block_bottom:
            # Table covers the entire block, replace the block with the table
            _blocks.append({'type': 'table', 'bbox': table_bbox})
        elif table_top > block_top and table_top < block_bottom:
            # Table overlaps the bottom of the block, adjust the bottom
            _blocks.append({'type': 'text', 'bbox': [0, block_top, width, table_top]})
            _blocks.append({'type': 'table', 'bbox': table_bbox})
        elif table_bottom > block_top and table_bottom < block_bottom:
            # Table overlaps the top of the block, adjust the top
            _blocks.append({'type': 'table', 'bbox': table_bbox})
            _blocks.append({'type': 'text', 'bbox': [0, table_bottom, width, block_bottom]})
        else:
            # Table does not affect this block
            _blocks.append(block)
    return _blocks

def extract_content(page, page_layout):
    if isinstance(page, fitz.Page):
        pdf = pdfplumber.open(page.parent.name)
        page = pdf.pages[page.number]
        for block in page_layout:
            block['bbox'] = scale_bbox(block['bbox'], 2)
            
    # 根据上边界位置排序
    page_layout = sorted(page_layout, key=lambda x: x['bbox'][1])

    width, height = math.floor(page.width), math.floor(page.height)
    
    # 根据page的上下边界, 缩小页面范围
    header_bottom_border = get_header_bottom_border(page_layout)
    footer_top_border = get_footer_top_border(page_layout, height)
    
    _page_layout = [{'type': 'text', 'bbox': [0, header_bottom_border, width, footer_top_border]}]
    tables = [bbox_correct(block, [0, 0, width, height]) for block in page_layout if block['type'] == 'table']
    for table in tables:
        _page_layout = split_page_layout_by_table(_page_layout, table, width, height)
    page_content = ""
    for block in _page_layout:
        if block['type'] == 'text':
            text = extract_text(page, block['bbox'])
            page_content += text + "\n\n"
        elif block['type'] == 'table':
            table = block.get('content') or extract_table(page, block['bbox'])
            # 清除Cell中的\n
            if table:
                cleaned_table = [[item.replace('\n', '') if isinstance(item, str) else item for item in row] for row in table]
                page_content += table_to_markdown(cleaned_table) + "\n\n"
    return page_content

def table_to_markdown(table):
    """
    Convert a 2D list to a Markdown table.

    :param table: 2D list containing the table data
    :return: String of the table in Markdown format
    """
    markdown_table = []

    # Process the header
    header = table[0]  # Assuming the second row is the header
    markdown_table.append("| " + " | ".join([str(cell) if cell else "" for cell in header]) + " |")
    markdown_table.append("|" + "---|" * len(header))

    # Process the rest of the rows
    for row in table[1:]:
        markdown_table.append("| " + " | ".join([str(cell) if cell else "" for cell in row]) + " |")

    return "\n".join(markdown_table)

def ends_with_punct(text):
    return any(text.endswith(punct) for punct in ['。', '!', '?'])

def match_title(text):
    chinese_brackets = title_patterns.get('chinese_brackets')
    arabic_brackets = title_patterns.get('arabic_brackets')
    chinese_number_comma = title_patterns.get('chinese_number_comma')
    arabic_number_comma = title_patterns.get('arabic_number_comma')
    arabic_number_dot = title_patterns.get('arabic_number_dot')
    arabic_dot_level = title_patterns.get('arabic_dot_level')

    pattern = f"^({chinese_brackets}|{arabic_brackets}|{chinese_number_comma}|{arabic_number_comma}|{arabic_number_dot}|{arabic_dot_level})"
    return bool(re.match(pattern, text))

def process_text(text):
    lines = text.split('\n')
    merged_text = ""
    for i in range(len(lines)):
        line = lines[i].strip()
        # 检查是否为最后一行
        if i < len(lines) - 1:
            next_line = lines[i + 1].strip()
            if match_title(line):
                merged_text += line + "\n\n"
            elif ends_with_punct(line) or match_title(next_line):
                merged_text += line + "\n\n"
            else:
                merged_text += line
        else:
            merged_text += line
    return merged_text

#=========================================
#         处理提取页面内容后的文本
#=========================================
import cn2an
# 直接返回阿拉伯数字或层级结构
def convert_number(text):
    if re.match(r"^\d+(\.\d+)*$", text):
        return text
    return cn2an.cn2an(text)

# 提取并转换数字
def extract_title_numbers(text):
    pattern = re.compile(r"([一二三四五六七八九十百千万]+|\d+(\.\d+)*)")
    matches = pattern.findall(text)
    numbers = [convert_number(matches[0][0])]
    return numbers

from anytree import Node        
class Tree:
    def __init__(self, chapter) -> None:
        self.root = Node(chapter, type="root", number=-1)
        self.pointer = self.root
        
    def add_node(self, title):
        node = self.build_node(title)
        while True:
            if self.pointer.is_root:
                logger.debug('at root', node)
                node.parent = self.root
                self.pointer = node
                break
            relationship = self.get_relationship(node)
            logger.debug(f'{node} is {self.pointer}\'s {relationship}')
            if relationship == "child":
                node.parent = self.pointer
                self.pointer = node
                break
            elif relationship == "sibling":
                node.parent = self.pointer.parent
                self.pointer = node
                break
            else:
                self.pointer = self.pointer.parent
        return node
    
    def get_relationship(self, node):
        try:
            if self.pointer.type == node.type and node.type == "arabic_dot_level":
                pointer_parts = self.pointer.number.split('.')
                node_parts = node.number.split('.')
                if len(node_parts) == len(pointer_parts) + 1 and node_parts[:-1] == pointer_parts:
                    return "child"
                elif len(node_parts) == len(pointer_parts) and node_parts[:-1] == pointer_parts[:-1] and node_parts[-1] != pointer_parts[-1]:
                    return "sibling"
                else:
                    return "uncertain"
            else:
                if self.pointer.type == node.type:
                    if int(node.number) == int(self.pointer.number) + 1:
                        return "sibling"
                else:
                    if int(node.number) == 1:
                        return "child"
        except Exception as e:
            # print(e)
            return "uncertain"
    
    def build_node(self, title):        
        for pattern_name, pattern in title_patterns.items():
            pattern = f"^{pattern}"
            match = re.match(pattern, title)
            if match:
                return Node(title, type=pattern_name, number=extract_title_numbers(title)[0])
        raise Exception(f"Build Node Failed, {title} not match any pattern")
    
    def print(self, node=None, prefix="", is_last=True, is_root=True):
        if node is None:
            node = self.root
        
        if is_root:
            print(node.name + f" (type={node.type}, number={node.number})")
        else:
            connector = "└── " if is_last else "├── "
            print(prefix + connector + node.name + f" (type={node.type}, number={node.number})")
            prefix += "    " if is_last else "│   "
        
        for i, child in enumerate(node.children):
            is_last_child = i == len(node.children) - 1
            self.print(child, prefix, is_last_child, is_root=False)
            
#=========================================
#          获取"管理层讨论与分析"
#=========================================
def chapter_range(chapters, keyword):
    if not chapters: raise Exception("No chapters found")
    
    df = pd.DataFrame(chapters)
    index = df[df['chapter'].str.contains(keyword)].index

    if len(index) == 0: raise Exception(f"No chapter contains '{keyword}'")
    
    start_index = index[0]
    start_page = df.loc[start_index, 'page']
    start_title = df.loc[start_index, 'chapter']
    
    # 判断是否有下一行
    if start_index + 1 < len(df):
        end_page = df.loc[start_index + 1, 'page']
        end_title = df.loc[start_index + 1, 'chapter']
    else:
        end_page = None
        end_title = None
    
    return start_page, end_page, start_title, end_title

def find_header_and_footer(path, start_page, end_page):
    def extract_layout(page):
        layout = recongize_layout(crop_page(page, scale_factor=1))
        headers = [block for block in layout if block['type'] == 'header']
        footers = [block for block in layout if block['type'] == 'footer']
        return headers, footers

    with fitz.open(path) as pdf:
        start_page_obj = pdf[int(start_page) - 1]
        end_page_obj = pdf[int(end_page) - 1]

        start_headers, start_footers = extract_layout(start_page_obj)
        end_headers, end_footers = extract_layout(end_page_obj)

        width, height = start_page_obj.mediabox_size
        page_size = (int(width), int(height))

        if start_headers == end_headers and start_footers == end_footers:
            return start_headers + start_footers, page_size
        else:
            return [], page_size

def extract_content_between_titles(page_content, start_title, end_title):
    """
    从 page_content 中提取位于 start_title 和 end_title 之间的内容。

    参数:
        page_content (str): 原始内容。
        start_title (str): 起始标题。
        end_title (str): 结束标题。

    返回:
        str: 提取的内容。
    """
    # 找到起始标题的位置
    start_position = page_content.find(start_title)
    if start_position == -1:
        raise ValueError(f"Start title '{start_title}' not found in the content.")
    
    # 提取从起始标题开始的内容
    page_content = page_content[start_position:]
    
    # 找到结束标题的位置
    end_position = page_content.find(end_title)
    if end_position != -1:
        page_content = page_content[:end_position]
    
    return page_content

def debug_page(path, page_num, page_layout):
    _page_layout = []
    for rect in page_layout:
        if rect['type'] == 'table': 
            _page_layout.append(rect)
            
    with fitz.open(path) as pdf:
        page = pdf[page_num]
        image = crop_page(page, scale_factor=1)
        debug_layout(image, _page_layout)

def read_management_chapter(path):
    chapters = extract_chapters(path)
    keywords = ["管理层讨论与分析", "董事会报告"]
    for keyword in keywords:
        try:
            start_page, end_page, start_title, end_title = chapter_range(chapters, keyword)
            break
        except:
            continue
    
    result = []
    page_content = ""
    tree = Tree(start_title)
    
    hnf, page_size = find_header_and_footer(path, start_page, end_page) # header and footer
    with pdfplumber.open(path) as pdf:
        for page_num in range(start_page, end_page + 1): # range的特点,不+1的话会少一页
            page = pdf.pages[page_num - 1]
            tables = page.find_tables()
            # 是否使用hnf模板
            page_layout = hnf[:] if (int(page.width), int(page.height)) == page_size else []
            
            if tables:
                for table in tables:
                    bbox = table.bbox
                    bbox = [bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1] # 扩大才能识别全
                    page_layout.append({'type': 'table', 'bbox': bbox, 'content': table.extract()})

            page_content += extract_content(page, page_layout)
            
    page_content = extract_content_between_titles(page_content, start_title, end_title)
    lines = page_content.split("\n")
    for line in lines:
        if match_title(line):
            node = tree.add_node(line)
            prefix = ("#" * node.depth + " ") if node.depth in range(1, 4) else ""
            result.append(prefix + node.name)
        else:
            result.append(line)
            
    return "\n".join(result)