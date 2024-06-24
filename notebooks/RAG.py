import os, sys
sys.path.append('..')

# import required dependencies
from langchain import hub

from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import chainlit as cl
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.embeddings import GPT4AllEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from utils.CustomMarkdownHeaderTextSplitter import CustomMarkdownHeaderTextSplitter
from utils.KeywordRetriever import KeywordRetriever

def build_retriever(vectorstore):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    parent_splitter = CustomMarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on[0:1])
    child_splitter = CustomMarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    keyword_retriever = KeywordRetriever(vectorstore=vectorstore)
    ensemble_retriever = EnsembleRetriever(retrievers=[retriever, keyword_retriever])
    return ensemble_retriever, retriever

# Set up RetrievelQA model
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-mistral")
DB_PATH = "vectorstores/db/600418"
print("-" * 50)
print(QA_CHAIN_PROMPT)
print("-" * 50)
model_name = "llama2-chinese:13b"

def qa_chain():
    llm = Ollama(model=model_name, verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    model_kwargs = {'device': 'cuda'}
    embeddings = HuggingFaceEmbeddings(model_name="../ai_models/m3e-base", model_kwargs=model_kwargs)
    vectorstore = Chroma(persist_directory=DB_PATH, collection_name="split_parents", embedding_function=embeddings)

    ensemble_retriever, retriever = build_retriever(vectorstore)
    chain = RetrievalQA.from_chain_type(
        llm,
        # retriever=vectorstore.as_retriever(),
        retriever=ensemble_retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,
    )
    return chain

@cl.on_chat_start
async def start():
    chain = qa_chain()
    msg = cl.Message(content="Firing up the research info bot...")
    await msg.send()
    msg.content = "Hi, welcome to research info bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])

    answer_source = ""
    sources = res["source_documents"]

    if sources:
        # Create a Markdown table header
        markdown_table = "| Header 1 | Content |\n|----------|---------|\n"
        for source in sources:
            print(source)
            page = source.metadata['Header 1']
            content = source.page_content.replace("\n", " ")
            markdown_table += f"| {page} | {content} |\n"
        answer_source += f"\n## Sources\n{markdown_table}"
    else:
        answer_source += f"\nNo Sources found"

    await cl.Message(content=answer_source).send()
