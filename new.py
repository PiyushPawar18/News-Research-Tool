import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from groq.api import GroqClient  # Replace OpenAI with Groq API

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("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")

# Initialize Groq client
groq_client = GroqClient(api_key=GROQ_API_KEY)

# Streamlit UI setup
st.title("RockyBot: News Research Tool \U0001F4C8")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_groq.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    try:
        # Load data
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading...Started...\u2705\u2705\u2705")
        data = loader.load()

        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...\u2705\u2705\u2705")
        docs = text_splitter.split_documents(data)

        # Create embeddings using Groq API
        embeddings = OpenAIEmbeddings(client=groq_client)  # Adapted to Groq
        vectorstore_groq = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...\u2705\u2705\u2705")
        time.sleep(2)

        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_groq, f)

        main_placeholder.text("Processing complete! FAISS index saved.")
    except Exception as e:
        st.error(f"Error occurred: {e}")

query = main_placeholder.text_input("Question:")
if query:
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=groq_client,  # Using Groq client
                    retriever=vectorstore.as_retriever()
                )
                result = chain({"question": query}, return_only_outputs=True)

                # Display answer
                st.header("Answer")
                st.write(result["answer"])

                # Display sources, if available
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")  # Split sources by newline
                    for source in sources_list:
                        st.write(source)
    except Exception as e:
        st.error(f"Error occurred: {e}")
