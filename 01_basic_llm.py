import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is not set")


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)


response = llm.invoke(
    "Explain what RAG is in simple terms."
)


print(response)
print("\n--- AI CONTENT ---")
print(response.content)