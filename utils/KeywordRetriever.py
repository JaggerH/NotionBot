from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.stores import BaseStore


class KeywordRetriever(BaseRetriever):
    """A keyword retriever that contains the top k documents that contain the user query.

    This retriever only implements the sync method _get_relevant_documents.

    If the retriever were to involve file access or network access, it could benefit
    from a native async implementation of `_aget_relevant_documents`.

    As usual, with Runnables, there's a default async implementation that's provided
    that delegates to the sync implementation running on another thread.
    
    Usage Example:
    keyword_retriever = KeywordRetriever(vectorstore=vectorstore)
    keyword_retriever.invoke("公司从事的业务情况")
    """

    store: BaseStore[str, Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        keywords = [query] # 为什么此处用list，因为我考虑后期通过分词扩展
        # where_document_condition = {"$or": [{"$contains": keyword} for keyword in keywords]} if len(keywords) > 1 else {"$contains": keywords[0]}
        # results = self.store.get(where_document=where_document_condition)

        keys = list(self.store.yield_keys())
        documents = self.store.mget(keys)
        result = []
        for doc in documents:
            if any(keyword in doc.page_content for keyword in keywords):
                result.append(doc)
                
        return result

    # Optional: Provide a more efficient native implementation by overriding
    # _aget_relevant_documents
    # async def _aget_relevant_documents(
    #     self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    # ) -> List[Document]:
    #     keywords = [query]
    #     where_document_condition = {"$or": [{"$contains": keyword} for keyword in keywords]} if len(keywords) > 1 else {"$contains": keywords[0]}
    #     results = self.vectorstore.get(where_document=where_document_condition)
        
    #     return [
    #         Document(page_content=document, metadata=metadata or {})
    #         for document, metadata in zip(results.get("documents", []), results.get("metadatas", []))
    #     ]