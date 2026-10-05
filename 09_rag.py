import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


load_dotenv()


# --------------------------------------------------
# 1. LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)


# --------------------------------------------------
# 2. Embeddings
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 3. Vector Store
# --------------------------------------------------

vector_store = Chroma(
    collection_name="company_policy",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)


# --------------------------------------------------
# 4. Retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# --------------------------------------------------
# 5. Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
    You are a company policy assistant.

    Answer the question using ONLY the provided context.

    If the answer cannot be found in the context,
    say "I don't know based on the provided document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)


# --------------------------------------------------
# 6. Helper function
# --------------------------------------------------

def format_documents(documents):
    return "\n\n".join(
        document.page_content
        for document in documents
    )


# --------------------------------------------------
# 7. RAG Chain
# --------------------------------------------------

rag_chain = (
    {
        "context": retriever | format_documents,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# --------------------------------------------------
# 8. Ask questions
# --------------------------------------------------

while True:

    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        break

    answer = rag_chain.invoke(question)

    print("\nAnswer:")
    print(answer)