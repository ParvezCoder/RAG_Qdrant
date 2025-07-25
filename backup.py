# 100% ok without streamlit (using Hugging face and Qdrant)
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant  import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")


embed_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

loader = PyPDFLoader("resume 2025.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 150
)
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant = QdrantVectorStore.from_documents(
    pages,
    embed_model,
    url = "https://3e4b9ca6-7b60-470d-9210-d0f2903b8970.eu-west-2-0.aws.cloud.qdrant.io:6333",
    api_key = qdrant_api_key,
    collection_name ="demooo",
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=qdrant.as_retriever()
)

user_query = input("enter your query: ")
response = qa_chain.invoke(user_query)
print(response)