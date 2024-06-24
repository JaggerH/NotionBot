from typing import List, Optional, Tuple

from langchain_core.documents import Document

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.text_splitter import TextSplitter

class CustomMarkdownHeaderTextSplitter(MarkdownHeaderTextSplitter, TextSplitter):
    """
    Usage Example:
    from utils.CustomMarkdownHeaderTextSplitter import CustomMarkdownHeaderTextSplitter

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    text_splitter = CustomMarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    documents = text_splitter.split_documents(docs)
    """
    def __init__(
        self,
        headers_to_split_on: List[Tuple[str, str]],
        return_each_line: bool = False,
        strip_headers: bool = False,
    ):
        """Create a new MarkdownHeaderTextSplitter.
        
        Args:
            headers_to_split_on: Headers we want to track
            return_each_line: Return each line w/ associated headers
            strip_headers: Strip split headers from the content of the chunk
        """
        TextSplitter.__init__(self)
        MarkdownHeaderTextSplitter.__init__(self, headers_to_split_on, return_each_line, strip_headers)
        
    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for text in texts:
            chunks = self.split_text(text)
            for chunk in chunks:
                chunk.metadata  = {**_metadatas[0], **chunk.metadata}
            documents.extend(chunks)
        return documents