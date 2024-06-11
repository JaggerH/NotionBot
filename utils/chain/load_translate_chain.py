from typing import Any, Mapping, Optional, Protocol

from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts import PromptTemplate

from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

prompt_template = """将以下内容翻译成中文

"{text}"

翻译内容："""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

def load_translate_chain(
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