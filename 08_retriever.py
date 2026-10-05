from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = Chroma(
    collection_name="company_policy",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)


retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)


question = "How many annual paid leaves do employees get?"


documents = retriever.invoke(question)


print("Retrieved documents:", len(documents))


for i, document in enumerate(documents):
    print("\n====================")
    print(f"RESULT {i + 1}")
    print("====================")
    print(document.page_content)
    print("Metadata:", document.metadata)