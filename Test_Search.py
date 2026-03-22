from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Load the existing ChromaDB — no re-indexing
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Ask a question
question = "How many vacation days do I get?"
results = retriever.invoke(question)

print(f"Question: {question}")
print(f"\nFound {len(results)} relevant chunks:\n")
for i, doc in enumerate(results):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content)
    print(f"Source: {doc.metadata}")
    print()