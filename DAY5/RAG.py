import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile

# --- Hardcoded Google API Key (for testing only) ---
GOOGLE_API_KEY = "AIzaSyCWjDvfquO7Yf6cOrnmC9xNE-1X_jVZiaM"

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Gemini RAG App", layout="wide")
st.title("📄🔍 RAG App with Google Gemini + HuggingFace Embeddings")

# --- Initialize Conversation Memory ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory = st.session_state.memory

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# User input
query = st.text_input("Ask a question about your document:")

# Process only if file and query exist
if uploaded_file and query:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Step 1: Load PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Step 2: Chunk the docs
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Step 3: Embeddings + FAISS vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Step 4: Gemini LLM setup
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7
    )

    # Step 5: Prompt Template (Optional, for chain consistency)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions about the provided documents."),
        ("human", "{question}")
    ])

    # Step 6: Conversational Retrieval Chain with memory
    retriever = vectorstore.as_retriever()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False
    )

    try:
        response = chain.run(query)
        st.success("Here is your Answer:")
        st.write(response)
    except Exception as e:
        st.error(f"Error: {e}")

elif uploaded_file:
    st.info("Please enter a question.")
elif query:
    st.info("Please upload a PDF document.")
