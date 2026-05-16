from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

texts = [
    "This website provides web services and digital solutions.",
    "We help businesses build scalable products and platforms.",
    "Our services include product management, AI solutions, and analytics."
]

embedding = OpenAIEmbeddings()

vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embedding,
    persist_directory="db"
)

retriever = vectorstore.as_retriever()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)
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
