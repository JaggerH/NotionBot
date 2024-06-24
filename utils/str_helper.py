from bs4 import BeautifulSoup

def html_table_to_array(html):
    # Parse the HTML
    soup = BeautifulSoup(html, 'html.parser')

    # Find all rows in the table
    rows = soup.find_all('tr')

    # Initialize a list to hold the table data
    table_data = []
    row_spans = []

    # Initialize a list for rowspan placeholders
    for _ in range(len(rows[0].find_all(['td', 'th']))):
        row_spans.append(0)

    # Iterate over the rows
    for row in rows:
        row_data = []
        cells = row.find_all(['td', 'th'])
        col_index = 0

        for cell in cells:
            # Skip columns that are spanned by the previous row
            while col_index < len(row_spans) and row_spans[col_index] > 0:
                row_data.append('')
                row_spans[col_index] -= 1
                col_index += 1
            
            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))
            value = cell.get_text(strip=True)

            # Append the cell value with the appropriate colspan
            for _ in range(colspan):
                row_data.append(value)
                if rowspan > 1:
                    row_spans[col_index] = rowspan - 1
                col_index += 1

        # Adjust for any remaining row spans
        while col_index < len(row_spans):
            if row_spans[col_index] > 0:
                row_data.append('')
                row_spans[col_index] -= 1
            col_index += 1
        
        table_data.append(row_data)

    return table_data


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

def table_to_markdown(table):
    """
    Convert a 2D list to a Markdown table.

    :param table: 2D list containing the table data
    :return: String of the table in Markdown format
    """
    markdown_table = []

    # Process the header
    header = table[0]
    markdown_table.append("| " + " | ".join([str(cell) if cell else "" for cell in header]) + " |")
    markdown_table.append("|" + "---|" * len(header))

    # Process the rest of the rows
    for row in table[1:]:
        markdown_table.append("| " + " | ".join([str(cell) if cell else "" for cell in row]) + " |")

    return "\n".join(markdown_table)

def format_documents_vscode(documents):
    """
    Used to print documents in vscode
    """
    separator_in_page = '-' * 50
    separator = "=" * 50

    for doc in documents:
        page_content = doc.page_content
        metadata = doc.metadata
        
        # Format metadata
        # formatted_metadata = f"Metadata:\nPage: {metadata['page']}\nSource: {metadata['source']}"
        formatted_metadata = str(metadata)
        
        # Format page content with line breaks
        formatted_content = "\n".join(page_content.split("\n"))
        
        # Print formatted content and metadata
        print("Page Content:\n" + formatted_content)
        print(separator_in_page)
        print(formatted_metadata)
        print(separator + "\n")