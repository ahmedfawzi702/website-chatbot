import streamlit as st
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain



st.set_page_config(
    page_title="🌐 Chat with Websites",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.title("🌐 Chat with Websites")



def get_website_content(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    texts = [doc.page_content for doc in chunks]

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)

    return vectorstore



def get_context_chain(vectorstore):
    llm = Ollama(model="yi:9b-chat")
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("system", "Given the above conversation and context, answer the question precisely.")
    ])

    return create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=prompt
    )


def get_conversation_rag_chain(retriever_chain):
    llm = Ollama(model="yi:9b-chat")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)



def get_response(user_question):
    if "retriever_chain" not in st.session_state:
        context = get_context_chain(st.session_state.document)
        st.session_state.retriever_chain = get_conversation_rag_chain(context)

    response = st.session_state.retriever_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_question
    })

    return response["answer"]



with st.sidebar:
    st.header("⚙️ Website Configuration")
    url = st.text_input("🔗 Enter the website URL")



if not url:
    st.info("🔍 Please enter a website URL to start chatting.")
else:
 
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="👋 Hello! How can I assist you with the website?")
        ]

    if "document" not in st.session_state:
        with st.spinner("📡 Loading website content..."):
            st.session_state.document = get_website_content(url)

 
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    
    user_question = st.chat_input("💬 Ask me anything about the website...")

    if user_question:
      
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.chat_history.append(HumanMessage(content=user_question))

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                answer = get_response(user_question)
                st.markdown(answer)
                st.session_state.chat_history.append(AIMessage(content=answer))
