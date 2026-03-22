from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Step 1: Load all .txt files from the data/ folder
loader = DirectoryLoader(
    "data/",
    glob="**/*.txt",
    loader_cls=TextLoader
)
docs = loader.load()
print(f"Loaded {len(docs)} documents")

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

# Step 3: Embed and store in ChromaDB
vectorstore = Chroma.from_documents(
    chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="chroma_db"
)
print("ChromaDB populated!")