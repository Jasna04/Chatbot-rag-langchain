from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load PDF
loader = PyPDFLoader("data/your_docs.pdf")
documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OpenAIEmbeddings()

# Store in FAISS
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")

print("✅ RAG Index created!")
