from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


text = "Employees receive 18 days of annual paid leave."


vector = embeddings.embed_query(text)


print("Vector type:", type(vector))
print("Vector dimensions:", len(vector))

print("\nFirst 10 values:")
print(vector[:10])