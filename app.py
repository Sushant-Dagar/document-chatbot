"""
Document Chatbot - A Streamlit-based document Q&A application
Uses Google Gemini API for embeddings and ChromaDB for vector storage
"""

import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import docx
import tempfile
import hashlib

load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Google Gemini embedding function
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=GOOGLE_API_KEY,
    model_name="models/embedding-001"
) if GOOGLE_API_KEY else None


def get_or_create_collection(collection_name: str):
    """Get or create a ChromaDB collection."""
    return chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=google_ef
    )


def extract_text_from_pdf(file) -> str:
    """Extract text content from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(file) -> str:
    """Extract text content from a DOCX file."""
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def extract_text_from_txt(file) -> str:
    """Extract text content from a TXT file."""
    return file.read().decode("utf-8")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Split text into overlapping chunks for better retrieval."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def generate_document_id(content: str) -> str:
    """Generate a unique ID for a document chunk."""
    return hashlib.md5(content.encode()).hexdigest()


def add_document_to_collection(collection, text: str, filename: str):
    """Add document chunks to the ChromaDB collection."""
    chunks = chunk_text(text)
    ids = [f"{filename}_{i}_{generate_document_id(chunk)}" for i, chunk in enumerate(chunks)]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    return len(chunks)


def query_documents(collection, query: str, n_results: int = 3) -> list:
    """Query the collection for relevant document chunks."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results


def generate_response(query: str, context: str) -> str:
    """Generate a response using Gemini API based on the query and context."""
    model = genai.GenerativeModel('gemini-pro')

    prompt = f"""You are a helpful assistant that answers questions based on the provided document context.

Context from documents:
{context}

User Question: {query}

Please provide a comprehensive answer based on the context above. If the answer cannot be found in the context, say so clearly."""

    response = model.generate_content(prompt)
    return response.text


def main():
    st.set_page_config(
        page_title="Document Chatbot",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 Document Chatbot")
    st.markdown("Upload documents and ask questions about their content!")

    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Document Upload")

        if not GOOGLE_API_KEY:
            st.error("Please set GOOGLE_API_KEY in your .env file")
            st.stop()

        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

        if uploaded_files:
            collection = get_or_create_collection("documents")

            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    for uploaded_file in uploaded_files:
                        # Extract text based on file type
                        if uploaded_file.name.endswith(".pdf"):
                            text = extract_text_from_pdf(uploaded_file)
                        elif uploaded_file.name.endswith(".docx"):
                            text = extract_text_from_docx(uploaded_file)
                        else:
                            text = extract_text_from_txt(uploaded_file)

                        # Add to collection
                        num_chunks = add_document_to_collection(
                            collection, text, uploaded_file.name
                        )
                        st.success(f"✅ {uploaded_file.name}: {num_chunks} chunks indexed")

        st.divider()
        st.header("📊 Collection Stats")
        try:
            collection = get_or_create_collection("documents")
            count = collection.count()
            st.metric("Total Chunks", count)
        except:
            st.info("No documents indexed yet")

        if st.button("Clear All Documents"):
            try:
                chroma_client.delete_collection("documents")
                st.success("Collection cleared!")
                st.rerun()
            except:
                st.info("No collection to clear")

    # Main chat interface
    st.header("💬 Chat with your Documents")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    collection = get_or_create_collection("documents")

                    if collection.count() == 0:
                        response = "Please upload and process some documents first before asking questions."
                    else:
                        # Query for relevant chunks
                        results = query_documents(collection, prompt)

                        # Combine context from results
                        context = "\n\n".join(results["documents"][0]) if results["documents"] else ""

                        # Generate response
                        response = generate_response(prompt, context)

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
