from langchain_community.document_loaders import TextLoader


loader = TextLoader(
    "data/company_policy.txt",
    encoding="utf-8"
)


documents = loader.load()


print("Number of documents:", len(documents))

print("\n--- CONTENT ---")
print(documents[0].page_content)

print("\n--- METADATA ---")
print(documents[0].metadata)