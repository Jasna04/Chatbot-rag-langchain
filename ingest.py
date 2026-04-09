from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load file
loader = TextLoader("data/sample.txt")
documents = loader.load()

# Create embeddings
embedding = OpenAIEmbeddings()

# Store in vector DB
db = Chroma.from_documents(documents, embedding, persist_directory="db")

db.persist()

print("✅ Data ingested successfully")
