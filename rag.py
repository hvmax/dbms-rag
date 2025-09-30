# --- RAG Chatbot with Groq API and Streamlit UI ---

# This script creates a web interface for the RAG chatbot using Streamlit.
# It includes a chat history that persists across user interactions.

# --- 1. Import Necessary Libraries ---
import os
import torch
import traceback
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- App Configuration ---
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 Chat with Your Documents")
st.write("This chatbot uses your local text files to answer questions. It's powered by Groq for super-fast responses.")

# --- Configuration & Caching ---
# Use Streamlit's caching to load the model and index only once.
@st.cache_resource
def load_components():
    """Loads all the necessary components for the RAG pipeline."""
    # Configuration
    EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
    # Dynamically create the index path based on the model name
    FAISS_INDEX_PATH = f"faiss_index_{EMBEDDING_MODEL_NAME.replace('/', '_')}"

    # Initialize the embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'mps' if torch.backends.mps.is_available() else 'cpu'}
    )

    # Load the vector store if it exists, otherwise create it
    if os.path.exists(FAISS_INDEX_PATH):
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        # One-time setup
        st.info("No existing index found. Building a new one... This may take a few minutes.")
        loader = DirectoryLoader('./', glob="**/*.pdf", show_progress=True)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(texts, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        st.success("New index built and saved successfully!")
    
    return vector_store

@st.cache_resource
def get_qa_chain(_vector_store):
    """Initializes and returns the RAG chain."""
    # Connect to the Groq API (uses GROQ_API_KEY from st.secrets)
    llm = ChatGroq(
        temperature=0.1,
        model_name="openai/gpt-oss-120b", #OR use llama-3.3-70b-versatile
        api_key=st.secrets["GROQ_API_KEY"]
    )

    prompt_template = """
You are given context extracted from a PDF document. 
Your task is to answer the user's question based ONLY on the provided context. 
Do not use outside knowledge or assumptions. 
If the context does not contain enough information to answer, respond with: "I don’t know."

Instructions:
- Provide a clear, concise, and factual answer.
- At the end of your answer, include a confidence score from 0 to 1, where:
  - 1.0 means the answer is fully supported by the context,
  - 0.5 means the answer is partially supported or ambiguous,
  - 0.0 means there is no support for the answer.

Context:
{context}

Question:
{question}

Helpful Answer (with confidence score):
"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_vector_store.as_retriever(search_kwargs={'k': 6}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# --- Load components and initialize the chain ---
try:
    vector_store = load_components()
    qa_chain = get_qa_chain(vector_store)
except Exception as e:
    st.error("There was an error initializing the chatbot. Please check your setup and API keys.")
    st.error(e)
    st.stop()

# --- Chat History Management ---
# Use Streamlit's session_state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and Chat Logic ---
if prompt := st.chat_input("Ask a question about your articles..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = qa_chain.invoke(prompt)
                st.markdown(response['result'])
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response['result']})
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})




