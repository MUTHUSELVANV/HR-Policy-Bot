from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Load existing ChromaDB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Prompt that prevents hallucination
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an HR assistant. Answer using ONLY the context below.
If the answer is not in the context, say:
"I don't have that information in the HR policies provided."

Context: {context}
Question: {question}
Answer:"""
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Modern way to build a RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

# Test questions
questions = [
    "How many vacation days do I get?",
    "Does parental leave apply to adoptive parents?",
    "Can I use company internet for personal use?",
    "What is the salary increment policy?"
]

for q in questions:
    print(f"\nQ: {q}")
    answer = chain.invoke(q)
    print(f"A: {answer}")
    print("-" * 50)