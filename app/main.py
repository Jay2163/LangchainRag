from fastapi import FastAPI
from pydantic import BaseModel
from .services.rag import create_rag_chain


app = FastAPI(
    title="LangChain Mini RAG"
)


rag_chain = create_rag_chain()


class QuestionRequest(BaseModel):

    question: str


@app.get("/")
def root():

    return {
        "message": "LangChain Mini RAG API"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    answer = rag_chain.invoke(
        request.question
    )

    return {
        "question": request.question,
        "answer": answer
    }