from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)


prompt = ChatPromptTemplate.from_template(
    """
    Explain {topic} in 3 sentences.
    """
)


parser = StrOutputParser()


chain = prompt | llm | parser


response = chain.invoke(
    {
        "topic": "RAG"
    }
)


print("res \n\n\n",response)
print("response type: ", type(response))