from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)


prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful technical teacher.

    Explain the following topic to a beginner:

    Topic: {topic}

    Give:
    1. Simple definition
    2. Technical explanation
    3. Real-world example
    """
)


chain = prompt | llm


response = chain.invoke(
    {
        "topic": "LangChain"
    }
)


print(response.content)