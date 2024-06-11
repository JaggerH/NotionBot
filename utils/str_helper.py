from bs4 import BeautifulSoup

def html_table_to_md(html):
    # 解析 HTML 表格
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')

    # 提取表头
    headers = [header.text for header in table.find_all('th')]

    # 提取表格内容
    rows = []
    for row in table.find_all('tr')[1:]:  # 跳过表头
        cells = row.find_all('td')
        row_data = [cell.text.strip() for cell in cells]
        rows.append(row_data)

    # 构建 Markdown 表格
    md_table = '| ' + ' | '.join(headers) + ' |\n'
    md_table += '| ' + ' | '.join(['---' for _ in headers]) + ' |\n'
    for row in rows:
        md_table += '| ' + ' | '.join(row) + ' |\n'

    return md_table