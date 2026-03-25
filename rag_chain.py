from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Load ChromaDB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Prompt with memory slot
PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an HR assistant. Answer using ONLY the context below.
If the answer is not in the context, say:
'I don't have that information in the HR policies provided.'

Context: {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Manual memory list
chat_history = []

def ask(question):
    # Get relevant docs
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # Build chain
    chain = PROMPT | llm | StrOutputParser()

    # Run with history
    answer = chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question
    })

    # Save to memory
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    return answer

# Multi-turn test
questions = [
    "What is the parental leave policy?",
    "Does it apply to adoptive parents too?",
    "How many weeks is primary caregiver parental leave?",
    "What about vacation days?"
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {ask(q)}")
    print("-" * 50)