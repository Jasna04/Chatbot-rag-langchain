# AI Chatbot (LangChain + FastAPI)

Simple chatbot using LangChain + OpenAI + ChromaDB (RAG).

## 🚀 Setup

1. Clone repo
2. Install dependencies:
   pip install -r requirements.txt

3. Set API key:
   export OPENAI_API_KEY=your_key

4. Run app:
   uvicorn app:app --reload

## 📡 API

POST /chat

{
  "message": "Your question"
}
