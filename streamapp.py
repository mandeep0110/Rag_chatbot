import streamlit as st
import os
import time
import tempfile
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit page setup
st.set_page_config(page_title="📄 DocBot - PDF Q&A", layout="wide")
st.title("🤖 DocBot - Ask Me Anything About Your PDF!")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload and process PDF
uploaded_file = st.file_uploader("📎 Upload your PDF file", type=["pdf"])

if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name}", icon="📄")

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("🔄 Processing PDF, generating embeddings..."):
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            # Text splitting
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(docs)

            # Embedding & Vector Store
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(split_docs, embeddings)

            # Save retriever in session
            st.session_state.vector_store = vector_store
            st.session_state.retriever = vector_store.as_retriever()

    st.info("📚 PDF processed! You can now ask questions.", icon="📘")

# Q&A Section
if st.session_state.retriever:
    user_input = st.chat_input("Ask something about the PDF...")

    if user_input:
        with st.spinner("🤖 Thinking..."):
            # LLM setup
            llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

            prompt = ChatPromptTemplate.from_template("""
                Answer the questions based on the provided context only.
                Please provide the most accurate response based on the question
                <context>
                {context}
                </context>

                Question: {input}
                Answer:
                """)

            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(st.session_state.retriever, document_chain)

            start_time = time.time()
            response = retrieval_chain.invoke({"input": user_input})
            end_time = time.time()

            answer = response["answer"]
            elapsed = round(end_time - start_time, 2)

            # Save to chat history
            st.session_state.chat_history.append({
                "question": user_input,
                "answer": answer,
                "response_time": elapsed
            })

            # Display answer
            with st.chat_message("assistant"):
                st.markdown("### 💬 Answer:")
                st.write(answer)
                st.markdown(f"🕒 Response Time: `{elapsed} sec`")

# Show Chat History
if st.session_state.chat_history:
    with st.expander("🧠 Chat History"):
        for i, item in enumerate(st.session_state.chat_history[::-1], 1):
            st.markdown(f"**Q{i}:** {item['question']}")
            st.markdown(f"**A{i}:** {item['answer']}")
            st.markdown(f"⏱️ *Answered in {item['response_time']} sec*")
            st.markdown("---")

# Footer
st.markdown("---")
st.caption("Created with ❤️ using LangChain, Groq, HuggingFace & Streamlit")
