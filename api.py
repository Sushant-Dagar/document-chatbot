"""
Flask REST API for Document Chatbot
Provides endpoints for document upload and query processing
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import docx
import tempfile
import hashlib
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Google Gemini embedding function
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=GOOGLE_API_KEY,
    model_name="models/gemini-embedding-001"
) if GOOGLE_API_KEY else None


def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_collection():
    """Get or create the documents collection."""
    return chroma_client.get_or_create_collection(
        name="documents",
        embedding_function=google_ef
    )


def extract_text(file_path: str) -> str:
    """Extract text from a document file."""
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join(page.extract_text() for page in reader.pages)
    elif file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def generate_id(content: str) -> str:
    """Generate unique ID for content."""
    return hashlib.md5(content.encode()).hexdigest()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "api_configured": bool(GOOGLE_API_KEY)
    })


@app.route('/upload', methods=['POST'])
def upload_document():
    """Upload and process a document."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Extract and process text
        text = extract_text(file_path)
        chunks = chunk_text(text)

        # Add to collection
        collection = get_collection()
        ids = [f"{filename}_{i}_{generate_id(chunk)}" for i, chunk in enumerate(chunks)]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )

        # Clean up uploaded file
        os.remove(file_path)

        return jsonify({
            "message": "Document processed successfully",
            "filename": filename,
            "chunks_created": len(chunks)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/query', methods=['POST'])
def query_documents():
    """Query the documents for answers."""
    data = request.get_json()

    if not data or 'query' not in data:
        return jsonify({"error": "Query is required"}), 400

    query = data['query']
    n_results = data.get('n_results', 3)

    try:
        collection = get_collection()

        if collection.count() == 0:
            return jsonify({
                "answer": "No documents have been uploaded yet. Please upload documents first.",
                "sources": []
            })

        # Query for relevant chunks
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        # Combine context
        context = "\n\n".join(results["documents"][0]) if results["documents"] else ""
        sources = [m["source"] for m in results["metadatas"][0]] if results["metadatas"] else []

        # Generate response using Gemini
        model = genai.GenerativeModel('gemini-flash-latest')
        prompt = f"""You are a helpful assistant that answers questions based on document context.

Context:
{context}

Question: {query}

Provide a comprehensive answer based on the context. If the answer cannot be found, say so clearly."""

        response = model.generate_content(prompt)

        return jsonify({
            "answer": response.text,
            "sources": list(set(sources)),
            "chunks_used": len(results["documents"][0]) if results["documents"] else 0
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/documents', methods=['GET'])
def list_documents():
    """List all indexed documents."""
    try:
        collection = get_collection()

        # Get all documents metadata
        results = collection.get()

        # Extract unique sources
        sources = set()
        for metadata in results.get("metadatas", []):
            if metadata and "source" in metadata:
                sources.add(metadata["source"])

        return jsonify({
            "total_chunks": collection.count(),
            "documents": list(sources)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/documents', methods=['DELETE'])
def clear_documents():
    """Clear all documents from the collection."""
    try:
        chroma_client.delete_collection("documents")
        return jsonify({"message": "All documents cleared successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
