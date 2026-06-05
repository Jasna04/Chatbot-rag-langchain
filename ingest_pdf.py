from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load PDF
loader = PyPDFLoader("pdfs/petpals.pdf")
documents = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)

# Embeddings
embedding = OpenAIEmbeddings()

# Connect to existing DB
vectorstore = Chroma(
    persist_directory="db",
    embedding_function=embedding
)

# Add PDF chunks
vectorstore.add_documents(docs)

print(f"Added {len(docs)} chunks from PDF")
