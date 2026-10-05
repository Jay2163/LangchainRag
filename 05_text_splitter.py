from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


print("Number of chunks:", len(chunks))


for i, chunk in enumerate(chunks):
    print("\n====================")
    print(f"CHUNK {i}")
    print("====================")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)