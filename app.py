# ===== monkey-patch sqlite3 to use the newer bundled implementation =====
import sys
try:
    import pysqlite3  # this must be in requirements.txt so it's available
    # Override the stdlib sqlite3 module with pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    # If this fails on Cloud it means pysqlite3-binary didn't install properly;
    # logs will show the failure. Fall back to whatever sqlite is available (likely too old).
    pass
# ======================================================================

import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

# Set HuggingFace embeddings
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI
st.title("PARZLY")
st.write("Upload PDF files and chat with their content.")

api_key = st.text_input("Enter your Groq API Key", type="password")

if api_key:
    llm = ChatGroq(model="gemma2-9b-It", groq_api_key=api_key)

    # Setup session state
    if "store" not in st.session_state:
        st.session_state.store = {}

    session_state = st.text_input("Enter a session ID", value="default_session")

    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

            # Remove temp file
            os.remove(temppdf)

        # Chunk and embed
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        texts = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(texts, embeddings)
        retriever = vectorstore.as_retriever()

        # Prompt templates
        contextualize_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [("system", contextualize_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            ["system", system_prompt, MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )

        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        retrieval_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        # Chat history session
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            retrieval_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Ask a question about the uploaded PDFs:")

        if user_input:
            session_history = get_session_history(session_state)

            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"session_id": session_state}
            )

            st.success(f"Answer: {response['answer']}")
            st.write("Chat History:", session_history.messages)

else:
    st.warning("Please enter your Groq API Key to continue.")
