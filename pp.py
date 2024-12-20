import os
import streamlit as st
import pickle
import time
import nltk
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
nltk.download('punkt')  # Ensure punkt is downloaded
nltk.download('averaged_perceptron_tagger')  # Download the missing resource

# Load GROQ API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit App UI
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# User input for URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():
        urls.append(url.strip())

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_groq.pkl"
main_placeholder = st.empty()

# Function to validate data loading
def validate_data(data):
    if not data:
        st.error("No data fetched. Please check the provided URLs.")
        st.stop()

# Function to validate document chunks
def validate_chunks(docs):
    if not docs:
        st.error("No document chunks created. Please verify the input data.")
        st.stop()

# Process URLs
if process_url_clicked:
    # Step 1: Load data
    st.info("Loading data from URLs...")
    loader = UnstructuredURLLoader(urls=urls)
    try:
        data = loader.load()
        validate_data(data)
        st.success("Data loading complete! ✅")
    except Exception as e:
        st.error(f"Error fetching or processing URLs: {e}")
        st.stop()
    
    # Step 2: Split data into chunks
    st.info("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)
    validate_chunks(docs)
    st.success(f"Text successfully split into {len(docs)} chunks! ✅")
    
    # Step 3: Create embeddings using Sentence Transformers
    st.info("Generating embeddings...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore_groq = FAISS.from_documents(docs, embeddings)
        st.success("Embeddings successfully generated! ✅")
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        st.stop()
    
    # Step 4: Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_groq, f)
        st.success("Embedding vectorstore saved successfully! ✅")

# Query Section
st.header("Ask a Question:")
query = st.text_input("Enter your question here:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=None,  # Replace with your preferred LLM, if any
                retriever=vectorstore.as_retriever()
            )
            try:
                st.info("Fetching results...")
                result = chain({"question": query}, return_only_outputs=True)
                # Display Results
                st.header("Answer")
                st.write(result["answer"])

                # Display Sources
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")
                    for source in sources_list:
                        st.write(source)
            except Exception as e:
                st.error(f"Error during query processing: {e}")
    else:
        st.error("No FAISS index found. Please process URLs first!")
