import os
import sys
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from paddleocr import PaddleOCR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cudnn_dir = os.path.join(os.path.dirname(sys.executable), r"Library\bin")
if cudnn_dir not in sys.path:
    os.environ["PATH"] = os.pathsep.join([os.environ["PATH"], cudnn_dir])
    sys.path.append(cudnn_dir)

def read_pdf_directory(path, strategy="fast"):
    """
    https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-pdf
    PDF可用的策略有"auto"、"hi_res"、"ocr_only"和"fast"。
    """
    unstructure_kwargs = {
        "strategy": strategy,  # fast模式不对全图片的PDF进行识别
        "languages": ["eng", "chi"]
    }
    loader = DirectoryLoader(path, glob="*.pdf", show_progress=True, use_multithreading=True, loader_cls=UnstructuredPDFLoader, loader_kwargs=unstructure_kwargs)
    return loader.load()

def ocr_pdf(path):
    """
    https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/docs/PP-StructureV2_introduction.md
    """
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
    result = ocr.ocr(path, cls=True)
    final_txt = [line[1][0] for res in result for line in res]
    return "\n".join(final_txt)

def auto_read_pdf(path):
    """
    函数的目的就是通过一个路径，直接返回PDF的docs
    """
    if os.path.isfile(path):
        print(f"{path} is a file.")
    else:
        # Read all PDFs in the directory using fast strategy
        documents = read_pdf_directory(path, strategy="fast")
        # List to store reprocessed documents
        reprocessed_documents = []

        # Iterate through documents to find and reprocess image-based PDFs
        for doc in documents:
            if doc.page_content.strip() == "":  # Check if the content is empty, indicating an image-based PDF
                text = ocr_pdf(doc.metadata["source"])
                reprocessed_doc = Document(page_content=text, metadata=doc.metadata)
                reprocessed_documents.append(reprocessed_doc)
            else:
                reprocessed_documents.append(doc)

        return reprocessed_documents
        