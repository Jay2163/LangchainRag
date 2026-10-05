from langchain_chroma import Chroma
from .embeddings import get_embeddings


def get_vector_store():

    embeddings = get_embeddings()

    return Chroma(
        collection_name="company_policy",
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )


def get_retriever():

    vector_store = get_vector_store()

    return vector_store.as_retriever(
        search_kwargs={
            "k": 3
        }
    )