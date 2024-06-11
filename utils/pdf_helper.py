import re
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import pdfplumber
from typing import Dict
from paddleocr import paddleocr, PaddleOCR
import logging
import cv2

paddleocr.logging.disable(logging.DEBUG)

# # 创建一个自定义日志记录器
logger = logging.getLogger('ttt')
# logger.setLevel(logging.ERROR)  # 设置日志级别

# # 创建控制台处理器并设置日志级别
# ch = logging.StreamHandler()
# ch.setLevel(logging.ERROR)

# # 创建格式化器并添加到处理器
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)

# # 将处理器添加到自定义日志记录器
# logger.addHandler(ch)

def find_pages_with_keyword(pdf_path, keyword, max_pages=20):
    # 打开PDF文件
    document = fitz.open(pdf_path)
    pages_with_keyword = []

    # 遍历每一页
    for page_num in range(min(len(document), max_pages)):
        page = document.load_page(page_num)
        text = page.get_text()
        if keyword in text:
            text_lines = text.split('\n')
            for line in text_lines:
                if line.replace(" ", "") == keyword:
                    pages_with_keyword.append(page_num + 1)  # 页码从1开始

    pages_with_keyword = list(set(pages_with_keyword))  # 去重
    return pages_with_keyword

def extract_text_from_page(pdf_path, page_number, header_height=0.1, footer_height=0.1, left=0, right=1):
    """
    从指定页面提取文本内容，并尝试过滤页头和页尾。

    参数:
        pdf_path (str): PDF 文件路径。
        page_number (int): 页码（从1开始）。
        header_height (float): 要过滤的页头高度比例（页面高度的百分比）。
        footer_height (float): 要过滤的页尾高度比例（页面高度的百分比）。

    返回:
        str: 过滤后的页面文本内容。
    """
    with pdfplumber.open(pdf_path) as pdf:
        if page_number < 1 or page_number > len(pdf.pages):
            return f"无效的页码：{page_number}"
        
        page = pdf.pages[page_number - 1]
        width, height = page.width, page.height

        # 定义要裁剪的区域
        cropping_box = (
            width * left,  # 左边界
            height * header_height,  # 上边界
            width * right,  # 右边界
            height * (1 - footer_height)  # 下边界
        )

        cropped_page = page.within_bbox(cropping_box)
        text = cropped_page.extract_text()
        
        return text

def extract_chapters(text, pdf_path, pages):
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
    if type(text) == list:
        text = "\n".join(text)
        
    chapters = []
    
    # 定义两种模式
    pattern_dots = re.compile(r'^(\S.*?)\s*\.+\s*(\d+)$', re.MULTILINE) # 第四节 公司治理....... 72
    pattern_page_at_start = re.compile(r'^\s*(\d+)\s+(\S.*)$', re.MULTILINE)

    # 6个点的正则
    pattern_dot_validate = re.compile(r'\.{6}')
    
    # 如果匹配到连续6个点的模式
    if pattern_dot_validate.search(text):
        logger.info("匹配到文本为模式一,形如\"第四节 公司治理....... 72\",采用pattern_dots模式")
        matches_dots = pattern_dots.findall(text)
        for chapter, start_page in matches_dots:
            start_page = int(start_page)
            if chapter.strip():
                chapters.append({"chapter": chapter.strip(), "start": start_page})
    # 如果匹配到页码在开头的模式
    else:
        logger.info(f'{pdf_path}异形文档')
        if len(pages) != 1: raise ValueError("异形文档必须只有一页")
        content_page_image = extract_image_from_pdf(pdf_path, pages[0])
        column = analyze_layout(content_page_image)
        if column == 2:
            logger.info(f'{pdf_path}文档左右布局')
            left_text = extract_text_from_page(pdf_path, pages[0], left=0, right=0.5)
            right_text = extract_text_from_page(pdf_path, pages[0], left=0.5, right=1)
            text = left_text + right_text
        matches_start = pattern_page_at_start.findall(text)
        for start_page, chapter in matches_start:
            start_page = int(start_page)
            if chapter.strip():
                chapters.append({"chapter": chapter.strip(), "start": start_page})
    
    return chapters

def extract_image_from_pdf(pdf_path, page_number):
    document = fitz.open(pdf_path)
    page = document.load_page(page_number - 1)
    pix = page.get_pixmap()
    image_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGRA2BGR)
    pix.save('temp.png')
    return image_data

def analyze_layout(image):
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

def get_bookmarks(filepath: str) -> Dict[int, str]:
    """
    读取 PDF 文件中的目录信息。

    参数:
        filepath (str): PDF 文件路径。

    返回:
        list of dict: 提取的章节信息，每个章节信息包含"chapter", "start", "end"。
    """
    bookmarks = []
    with fitz.open(filepath) as doc:
        toc = doc.get_toc()  # [[lvl, title, page, …], …]
        for level, title, page in toc:
            bookmarks.append({'chapter': title, 'start': page, 'level': level})
    return bookmarks
    
def extract_chapter_info_from_pdf(pdf_path, debug=True):
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
    chapters = get_bookmarks(pdf_path)
    if len(chapters) > 3: 
        logger.info(f"{pdf_path}自带标签符合条件")
        chapters = [item for item in chapters if item['level'] == 1] # 仅保留level==1的item
        return chapters

    keyword = '目录'
    pages = find_pages_with_keyword(pdf_path, keyword)
    if not pages: 
        logger.error(f"{pdf_path}未找到包含关键词'{keyword}'的页面。")
        return []
    logger.info(f"find_pages_with_keyword: {pdf_path}, pages: {pages}")
    
    pages_content = [extract_text_from_page(pdf_path, page) for page in pages]
    
    chapters = extract_chapters(pages_content, pdf_path, pages)
    return chapters