import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.title("HR Policy Assistant")
st.caption("Ask me anything about company policies.")

@st.cache_resource
def load_chain():
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an HR assistant. Answer using ONLY the context below.
If the answer is not in the context, say:
'I don't have that information in the HR policies provided.'
Context: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    return retriever, prompt, llm

retriever, prompt, llm = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask about HR policies..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "chat_history": st.session_state.chat_history,
        "question": question
    })

    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=answer))
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)