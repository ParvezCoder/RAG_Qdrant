import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Title and instructions
st.set_page_config(page_title="Q&A with Resume", layout="centered")
st.title("🤖 Resume Q&A App using Gemini + Qdrant")
st.markdown("Ask anything about the uploaded resume!")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file (like resume)", type="pdf")

if uploaded_file:
    # Save the file to disk
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split PDF
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(pages)

    # Embedding model
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Qdrant vector store setup
    qdrant = QdrantVectorStore.from_documents(
        docs,
        embed_model,
        url="https://3e4b9ca6-7b60-470d-9210-d0f2903b8970.eu-west-2-0.aws.cloud.qdrant.io:6333",
        api_key=qdrant_api_key,
        collection_name="demooo"
    )

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=google_api_key
    )

    # Chain setup
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=qdrant.as_retriever()
    )

    # User query input
    user_query = st.text_input("🔍 Ask a question about the resume:")

    if user_query:
        with st.spinner("Generating answer..."):
            response = qa_chain.invoke(user_query)
            st.success("✅ Answer:")
            st.write(response["result"])
