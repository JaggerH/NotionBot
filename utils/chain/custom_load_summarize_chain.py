from typing import Any, Mapping, Optional, Protocol

from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts import PromptTemplate

from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

prompt_template = """用中文简要概括以下内容:

"{text}"

简要总结:
<公司简称><股票代码><公告标题>
<公告主要内容>
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

def custom_load_summarize_chain(
    llm: BaseLanguageModel,
    prompt: BasePromptTemplate = PROMPT,
    document_variable_name: str = "text",
    verbose: Optional[bool] = None,
    **kwargs: Any,
) -> StuffDocumentsChain:
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)  # type: ignore[arg-type]
    # TODO: document prompt
    return StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name=document_variable_name,
        verbose=verbose,  # type: ignore[arg-type]
        **kwargs,
    )