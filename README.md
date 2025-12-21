# Document Chatbot

A full-stack document-based chatbot using Python, Streamlit, and Google Gemini API. Upload documents and ask questions about their content using natural language.

## Features

- **Document Upload**: Support for PDF, DOCX, and TXT files
- **Vector Search**: ChromaDB for efficient document retrieval
- **AI-Powered Responses**: Google Gemini API for embeddings and Q&A
- **REST API**: Flask microservices for document processing
- **Docker Support**: Easy deployment with Docker Compose

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Flask REST API
- **Vector DB**: ChromaDB
- **AI/ML**: Google Gemini API
- **Containerization**: Docker

## Quick Start

### Prerequisites

- Python 3.11+
- Google API Key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sushantdagar/document-chatbot.git
cd document-chatbot
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Running the Application

**Streamlit UI:**
```bash
streamlit run app.py
```

**Flask API:**
```bash
python api.py
```

**Docker:**
```bash
docker-compose up --build
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/upload` | POST | Upload document |
| `/query` | POST | Query documents |
| `/documents` | GET | List documents |
| `/documents` | DELETE | Clear all documents |

## Project Structure

```
document-chatbot/
├── app.py              # Streamlit application
├── api.py              # Flask REST API
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose setup
└── README.md           # Documentation
```

## License

MIT License
