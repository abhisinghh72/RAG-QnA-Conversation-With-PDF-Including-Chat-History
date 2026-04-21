import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


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
from langsmith import traceable


os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational RAG Chatbot With PDF uploads and chat history")
st.write("Upload PDFs and chat with their content")

api_key = st.text_input("Enter your Groq API key:", type="password")


if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="groq/compound")

    session_id = st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader(
        "Choose PDF files", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        documents = []

        for uploaded_file in uploaded_files:
            temppdf = "./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)


        # Split + Embeddings


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500
        )

        splits = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db" 
        )

        retriever = vectorstore.as_retriever()

        # History-aware retriever

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given a chat history and the latest user question, "
             "formulate a standalone question. Do NOT answer."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )


        # QA Chain

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Use the retrieved context to answer the question. "
             "If unknown, say you don't know. Max 3 sentences.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )


        # Memory


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

     
        # LANGSMITH TRACEABLE FUNCTION
   

        @traceable(name="Full RAG Pipeline")
        def run_query(chain, user_input, session_id):
            return chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id},
                    "run_name": "RAG Chat with PDF",
                    "tags": ["rag", "pdf", "chatbot"]
                }
            )

        # UI
  

        user_input = st.text_input("Your question:")

        if user_input:
            session_history = get_session_history(session_id)

            # USE TRACEABLE FUNCTION
            response = run_query(
                conversational_rag_chain,
                user_input,
                session_id
            )

            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)

else:
    st.warning("Please enter the Groq API Key")