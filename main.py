import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# 👉 Read API keys from Streamlit Secrets
google_api_key = st.secrets.get("GOOGLE_API_KEY")
qdrant_api_key = st.secrets.get("QDRANT_API_KEY")

# ✅ Check if API keys are available
if not google_api_key or not qdrant_api_key:
    st.error("❌ Missing API keys! Please set them in Streamlit → Settings → Secrets.")
    st.stop()

# ✅ Page layout
st.set_page_config(page_title="Q&A with Resume", layout="centered")
st.title("🤖 Resume Q&A App using Gemini + Qdrant")
st.markdown("Upload a resume (PDF), then ask questions about it using Gemini AI!")

# ✅ File uploader
uploaded_file = st.file_uploader("📄 Upload your resume (PDF only)", type="pdf")

if uploaded_file:
    # Save uploaded file
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.read())

    # Load and split PDF
    loader = PyPDFLoader(uploaded_file.name)
    pages = loader.load_and_split()

    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(pages)

    # Embedding model
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Setup Qdrant vector store
    with st.spinner("🔄 Indexing your resume in Qdrant..."):
        qdrant = QdrantVectorStore.from_documents(
            docs,
            embed_model,
            url="https://3e4b9ca6-7b60-470d-9210-d0f2903b8970.eu-west-2-0.aws.cloud.qdrant.io:6333",
            api_key=qdrant_api_key,
            collection_name="demooo"
        )

    # LLM model (Gemini)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=google_api_key
    )

    # Create the Retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=qdrant.as_retriever()
    )

