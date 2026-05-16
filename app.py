from fastapi import FastAPI
from pydantic import BaseModel

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Create FastAPI app
app = FastAPI()

# Request model
class ChatRequest(BaseModel):
    message: str

# Embeddings
embedding = OpenAIEmbeddings()

# Vector DB
vectorstore = Chroma
persist_directory="db"
embedding_function=embedding
)

# Retriever
retriever = vectorstore.as_retriever()

# LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# Home route
@app.get("/")
def home():
    return {"message": "Chatbot API running 🚀"}

# Chat route
@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response = qa_chain.invoke({
            "query": request.message
        })

        return {
            "reply": response["result"]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()

        return {"reply": str(e)}
