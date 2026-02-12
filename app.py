import streamlit as st
import os
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# -------------------------
# Load Environment
# -------------------------

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")


# -------------------------
# Streamlit UI
# -------------------------
st.title("📄 Chat with your PDF (Groq + RAG)")
a="Manish"
st.write(f'A simple Project By::{a}')
st.write("Upload a PDF and ask any questions about it.")

import os
api_key = os.getenv("GROQ_API_KEY")


if api_key:

    # -------------------------
    # LLM
    # -------------------------
    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.3-70b-versatile"
    )

    # -------------------------
    # Embeddings
    # -------------------------
    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # -------------------------
    # Session Handling
    # -------------------------
    session_id = ("Session ID")

    if "store" not in st.session_state:
        st.session_state.store = {}

    # -------------------------
    # File Upload
    # -------------------------
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type="pdf"
    )

    if uploaded_file:

        # Save temp file
        temp_pdf = "./temp.pdf"
        with open(temp_pdf, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load PDF
        loader = PyPDFLoader(temp_pdf)
        documents = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)

        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding
        )

        retriever = vectorstore.as_retriever()

        # -------------------------
        # Contextualize Question Prompt
        # -------------------------
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given a chat history and the latest user question "
             "which might reference context in the chat history, "
             "formulate a standalone question which can be understood "
             "without the chat history. DO NOT answer the question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
        )

        # -------------------------
        # QA Prompt
        # -------------------------
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an assistant for question-answering tasks. "
             "Use the following retrieved context to answer the question. "
             "If you don't know the answer, say you don't know.\n\n"
             "{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        # -------------------------
        # Chat History Function
        # -------------------------
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # -------------------------
        # Chat Input
        # -------------------------
        user_input = st.text_input("Ask a question about the PDF")

        if user_input:
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            st.write("### Answer:")
            st.write(response["answer"])
