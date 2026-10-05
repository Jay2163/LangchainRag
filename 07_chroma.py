from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


loader = TextLoader(
    "data/company_policy.txt",
    encoding="utf-8"
)

documents = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
)

chunks = splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="company_policy"
)


print("Documents stored successfully.")

print("Number of chunks:", len(chunks))