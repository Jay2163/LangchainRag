from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .llm import get_llm
from .vectorstore import get_retriever


def format_documents(documents):

    return "\n\n".join(
        document.page_content
        for document in documents
    )


def create_rag_chain():

    llm = get_llm()

    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_template(
        """
        You are a company policy assistant.

        Answer using ONLY the provided context.

        If the answer is not available,
        say you don't know.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    chain = (
        {
            "context": retriever | format_documents,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain