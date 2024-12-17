import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings  # Import SentenceTransformer

from apikey import GROQ_API_KEY  # Ensure this file contains a valid Groq API key
from groq import Groq  # Use the Groq API for language model queries

# Load API key
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Streamlit App UI
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_groq.pkl"

main_placeholder = st.empty()

# Initialize Groq for LLM
llm = Groq(api_key=GROQ_API_KEY)

if process_url_clicked:
   # Data Loading
loader = UnstructuredURLLoader(urls=[url for url in urls if url.strip()])
main_placeholder.text("Data Loading...Started...✅✅✅")
data = loader.load()

# Validate Data
if not data:
    st.error("No data fetched. Please check the provided URLs.")
    st.stop()

# Text Splitting
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', '.', ','],
    chunk_size=1000
)
main_placeholder.text("Text Splitter...Started...✅✅✅")
docs = text_splitter.split_documents(data)

# Validate Chunks
if not docs:
    st.error("No document chunks created. Please verify the input data.")
    st.stop()

# Print first few chunks for debugging
st.write(f"Number of document chunks: {len(docs)}")
for i, doc in enumerate(docs[:5]):
    st.write(f"Chunk {i+1}: {doc.page_content}")

# Embedding Generation
from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore_groq = FAISS.from_documents(docs, embeddings)
main_placeholder.text("Embedding Vector Started Building...✅✅✅")


# Query Section
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            
            # Display Results
            st.header("Answer")
            st.write(result["answer"])

            # Display Sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
