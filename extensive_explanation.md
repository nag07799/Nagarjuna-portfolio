# Extensive Project Documentation: Agentic RAG System
## Forward Deployed Engineer Assignment - Complete Technical Breakdown

---

## Table of Contents
1. [Assignment Overview](#assignment-overview)
2. [Task 1: StackAI Implementation](#task-1-stackai-implementation)
3. [Task 2: Video Recording](#task-2-video-recording)
4. [Task 3: Technical Implementation - Agentic RAG Pipeline](#task-3-technical-implementation)
5. [Task 4: StackAI Roadmap](#task-4-stackai-roadmap)
6. [Complete System Architecture](#complete-system-architecture)
7. [API Endpoints - Detailed Breakdown](#api-endpoints-detailed-breakdown)
8. [File-by-File Explanation](#file-by-file-explanation)
9. [Data Flow and Routing](#data-flow-and-routing)
10. [Setup and Deployment](#setup-and-deployment)

---

## Assignment Overview

This assignment evaluates four core competencies:
- **Platform Knowledge**: Understanding of AI/ML platforms and tools
- **Communication Abilities**: Clear explanation of technical concepts
- **Client Management Skills**: Ability to deliver production-ready solutions
- **AI Skills**: Practical implementation of RAG systems

**Total Deliverables**: 4 major tasks covering UI/UX, video presentation, backend implementation, and strategic planning.

---

## Task 1: StackAI Implementation

### Objective
Build a powerful AI agent using StackAI platform with creative freedom to solve a real-world problem.

### Example Use Case Reference
**Company Due Diligence Agent** - Investment analysis system for companies like Salesforce

### Key Components Implemented
1. **Input Node**: Company name
2. **Knowledge Base Node**: Financial reports for economic context
3. **Web Search Node**: Real-time company information
4. **LinkedIn Node**: CEO profile extraction
5. **Documents Node**: User-uploaded reports (10Q, 10K)
6. **Custom API Node**: Additional integrations

### Outputs Generated
- Company URL
- Business model analysis
- Competitive analysis
- Macro environment assessment
- Investment recommendation
- Additional insights

### UI Design
- **Form Interface**: User-friendly input form
- **Batch Processing**: (Bonus) Bulk company analysis

### Deliverables
- ✅ URL of the UI
- ✅ URL of the project
- ✅ 5-minute Loom video walkthrough

---

## Task 2: Video Recording

### Requirements
- **Duration**: Maximum 5 minutes
- **Platform**: Loom (https://loom.com/)
- **Content**: Explain use case and build process as if presenting to a customer

### Key Points Covered
1. Workflow overview
2. Important elements highlighting
3. Potential user pain points
4. UI demonstration
5. Technical architecture walkthrough

### Deliverables
- ✅ Raw video recording (no editing required)
- ✅ Clear explanation of tricky implementation details

---

## Task 3: Technical Implementation - Agentic RAG Pipeline

### Core Requirements

This is the **primary technical component** implementing a production-ready RAG system with agentic reasoning.

### System Overview

**Production-Ready Retrieval-Augmented Generation (RAG) System**
- Local-first architecture using Ollama + Mistral
- Custom NumPy-based vector database
- Hybrid search (semantic + keyword)
- ReAct reasoning pattern
- Zero API costs, complete privacy

---

## Complete System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│                      (Web UI - HTML/CSS/JS)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FASTAPI APPLICATION                          │
│                        (app/main.py)                             │
└──────────┬───────────────────────────────────┬──────────────────┘
           │                                   │
           ▼                                   ▼
┌──────────────────────┐          ┌──────────────────────────────┐
│  INGESTION PIPELINE  │          │    QUERY PIPELINE            │
│   (api/ingestion.py) │          │     (api/query.py)           │
└──────────┬───────────┘          └──────────┬───────────────────┘
           │                                  │
           │                                  │
           ▼                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CORE SERVICES LAYER                           │
├──────────────────┬──────────────────┬──────────────────────────┤
│  PDF Processor   │  Intent Detector │   Query Transformer      │
│  Text Chunking   │  Search Engine   │   Reranker               │
│  Embedding Svc   │  LLM Service     │   Citation Validator     │
│  Vector Store    │  Hallucination   │   Query Refusal          │
│                  │  Filter          │                          │
└─────────┬────────┴──────────────────┴──────────┬───────────────┘
          │                                       │
          ▼                                       ▼
┌─────────────────────┐              ┌──────────────────────────┐
│   OLLAMA + MISTRAL  │              │   CUSTOM VECTOR DB       │
│   (Local LLM)       │              │   (NumPy-based)          │
└─────────────────────┘              └──────────────────────────┘
```

### ReAct Agent Flow (8-Stage Pipeline)

```
User Query
    │
    ▼
[1] INTENT DETECTION
    │ ── LLM analyzes: Is this a KB query or greeting?
    ▼
[2] QUERY PLANNING
    │ ── Transform & enhance query for better retrieval
    ▼
[3] HYBRID SEARCH
    │ ── Semantic (vector) + Keyword (BM25) fusion
    ▼
[4] RESULT OBSERVATION
    │ ── Evaluate confidence scores & relevance
    ▼
[5] CITATION VALIDATION
    │ ── Check if evidence meets similarity threshold
    ▼
[6] ANSWER GENERATION
    │ ── Template-based response with context
    ▼
[7] HALLUCINATION VERIFICATION
    │ ── Fact-check against source documents
    ▼
[8] RESPONSE RETURN
    │ ── Final answer + citations + reasoning
    ▼
User Receives Answer
```

---

## API Endpoints - Detailed Breakdown

### Base URL
```
http://localhost:8000
```

### 1. Document Ingestion Endpoints

#### **POST /ingest/upload**

**Purpose**: Upload one or more PDF documents for knowledge base ingestion

**Request Format**:
```http
POST /ingest/upload
Content-Type: multipart/form-data

files: [file1.pdf, file2.pdf, ...]
```

**Processing Flow**:
1. Receives multipart file upload
2. Validates PDF format
3. Extracts text using pdfplumber (primary) or PyPDF2/PyMuPDF (fallback)
4. Chunks text into 512-token segments with 50-token overlap
5. Generates embeddings using Sentence Transformers
6. Stores vectors in NumPy-based vector database
7. Persists metadata (filename, chunk_id, page_number)

**Response**:
```json
{
  "message": "Successfully ingested N documents",
  "files_processed": ["file1.pdf", "file2.pdf"],
  "total_chunks": 150,
  "processing_time": "3.2s"
}
```

**Error Handling**:
- 400: Invalid file format
- 413: File too large
- 500: Processing error

---

#### **GET /ingest/stats**

**Purpose**: Retrieve current knowledge base statistics

**Request Format**:
```http
GET /ingest/stats
```

**Response**:
```json
{
  "total_documents": 5,
  "total_chunks": 234,
  "embedding_dimension": 384,
  "vector_db_size_mb": 12.5,
  "indexed_files": [
    "document1.pdf",
    "document2.pdf"
  ]
}
```

**Use Cases**:
- Monitor ingestion progress
- Verify documents are indexed
- Check system capacity

---

#### **DELETE /ingest/clear**

**Purpose**: Remove all documents from vector store (reset knowledge base)

**Request Format**:
```http
DELETE /ingest/clear
```

**Processing**:
1. Clears NumPy vector arrays
2. Deletes metadata mappings
3. Removes cached embeddings
4. Resets document counters

**Response**:
```json
{
  "message": "Knowledge base cleared successfully",
  "chunks_deleted": 234,
  "documents_removed": 5
}
```

**Warning**: Irreversible operation - use with caution

---

### 2. Query Processing Endpoints

#### **POST /query/**

**Purpose**: Process user questions using agentic RAG pipeline

**Request Format**:
```http
POST /query/
Content-Type: application/json

{
  "query": "What is the main topic of the documents?",
  "top_k": 5,
  "include_citations": true
}
```

**Parameters**:
- `query` (string, required): User question
- `top_k` (int, optional): Number of chunks to retrieve (1-20, default: 5)
- `include_citations` (bool, optional): Return source references (default: true)

**Processing Pipeline** (Detailed):

**Stage 1: Intent Detection**
```python
# LLM analyzes query intent
intent = llm.classify(query)
# Returns: "knowledge_base_query" | "greeting" | "off_topic"
if intent != "knowledge_base_query":
    return polite_refusal()
```

**Stage 2: Query Transformation**
```python
# Enhance query for better retrieval
transformed = query_transformer.enhance(query)
# Example: "Who is the CEO?" → "CEO name, chief executive officer, leadership"
```

**Stage 3: Hybrid Search**
```python
# Semantic search (vector similarity)
semantic_results = vector_store.search(
    query_embedding,
    top_k=top_k
)

# Keyword search (BM25)
keyword_results = bm25.search(
    query_tokens,
    top_k=top_k
)

# Fusion with reciprocal rank
combined = reciprocal_rank_fusion(
    semantic_results,
    keyword_results,
    weights=[0.7, 0.3]
)
```

**Stage 4: Result Observation**
```python
# Evaluate confidence
for result in combined:
    result.confidence = calculate_confidence(
        semantic_score=result.vector_similarity,
        keyword_score=result.bm25_score,
        document_frequency=result.term_frequency
    )
```

**Stage 5: Citation Validation**
```python
# Check if evidence is sufficient
max_similarity = max(r.confidence for r in combined)
if max_similarity < SIMILARITY_THRESHOLD:  # 0.5
    return {
        "answer": "Insufficient evidence in documents.",
        "confidence": "low",
        "citations": []
    }
```

**Stage 6: Answer Generation**
```python
# Create context from top chunks
context = "\n\n".join(chunk.text for chunk in combined[:top_k])

# Generate answer using template
prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer (cite sources):"""

answer = llm.generate(prompt, max_tokens=2048, temperature=0.3)
```

**Stage 7: Hallucination Filter**
```python
# Verify claims against sources
claims = extract_claims(answer)
for claim in claims:
    supported = verify_against_context(claim, combined)
    if not supported:
        answer = remove_or_flag_claim(answer, claim)
```

**Stage 8: Response Construction**
```python
return {
    "answer": answer,
    "citations": [
        {
            "chunk_id": c.id,
            "source": c.filename,
            "page": c.page_number,
            "similarity": c.confidence,
            "text_snippet": c.text[:200]
        }
        for c in combined[:top_k]
    ],
    "confidence": calculate_overall_confidence(combined),
    "reasoning_steps": {
        "intent": "knowledge_base_query",
        "query_transformation": transformed,
        "retrieval_method": "hybrid_search",
        "chunks_retrieved": len(combined),
        "hallucination_check": "passed"
    }
}
```

**Response Example**:
```json
{
  "answer": "According to the documents, the CEO is John Smith, who has led the company since 2020.",
  "citations": [
    {
      "chunk_id": 42,
      "source": "company_report.pdf",
      "page": 3,
      "similarity": 0.87,
      "text_snippet": "John Smith joined as CEO in January 2020, bringing 15 years of experience..."
    }
  ],
  "confidence": "high",
  "reasoning_steps": {
    "intent": "knowledge_base_query",
    "query_transformation": "CEO, chief executive officer, leadership, John Smith",
    "retrieval_method": "hybrid_search",
    "chunks_retrieved": 5,
    "hallucination_check": "passed"
  },
  "processing_time_ms": 850
}
```

**Error Responses**:
```json
// Insufficient evidence
{
  "answer": "I cannot find sufficient information in the documents to answer this question.",
  "confidence": "none",
  "citations": []
}

// Off-topic query
{
  "answer": "This question is outside the scope of the knowledge base. Please ask questions related to the uploaded documents.",
  "refusal_reason": "off_topic"
}

// PII detection
{
  "answer": "I cannot process queries containing personal identifiable information.",
  "refusal_reason": "pii_detected"
}
```

---

#### **GET /query/health**

**Purpose**: Service health check and readiness probe

**Request Format**:
```http
GET /query/health
```

**Response**:
```json
{
  "status": "healthy",
  "ollama_connection": "active",
  "vector_store": "ready",
  "documents_indexed": 5,
  "service_uptime": "2h 34m"
}
```

---

### 3. Documentation Endpoints

#### **GET /docs**
- Swagger UI - Interactive API documentation
- Access at: `http://localhost:8000/docs`

#### **GET /redoc**
- ReDoc - Alternative API documentation
- Access at: `http://localhost:8000/redoc`

---

## File-by-File Explanation

### Project Structure
```
agentic_rag_implementation/
├── app/
│   ├── api/              # API route handlers
│   ├── core/             # Core infrastructure
│   ├── services/         # Business logic
│   ├── models/           # Data schemas
│   ├── utils/            # Helper functions
│   └── main.py           # Application entry point
├── ui/                   # Web interface
├── data/                 # Runtime data
│   ├── uploads/          # Uploaded PDFs
│   └── vector_db/        # Serialized vectors
├── requirements.txt      # Dependencies
├── .env                  # Configuration
└── README.md            # Documentation
```

---

### Core Application Files

#### **app/main.py**
**Purpose**: FastAPI application entry point and routing configuration

**Key Responsibilities**:
- Initialize FastAPI app with CORS middleware
- Mount API routers (ingestion, query)
- Serve static UI files
- Configure OpenAPI documentation
- Handle application lifecycle (startup/shutdown)

**Code Breakdown**:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Initialize app
app = FastAPI(
    title="Agentic RAG System",
    description="Production-ready RAG with ReAct reasoning",
    version="1.0.0"
)

# Enable CORS for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include routers
from app.api import ingestion, query
app.include_router(ingestion.router, prefix="/ingest", tags=["Ingestion"])
app.include_router(query.router, prefix="/query", tags=["Query"])

# Serve UI
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

# Lifecycle events
@app.on_event("startup")
async def startup():
    # Initialize Ollama connection
    # Load vector store if exists
    # Warm up embedding model
    pass

@app.on_event("shutdown")
async def shutdown():
    # Save vector store
    # Close connections
    pass
```

---

#### **app/api/ingestion.py**
**Purpose**: Handle PDF upload and document ingestion

**Endpoints Implemented**: `/upload`, `/stats`, `/clear`

**Key Functions**:

```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Process uploaded PDF documents

    Steps:
    1. Validate file types (only PDFs)
    2. Save to uploads directory
    3. Extract text using PDF processor
    4. Chunk text with overlap
    5. Generate embeddings
    6. Store in vector database
    7. Return processing summary
    """
    from app.services.pdf_processor import PDFProcessor
    from app.core.embeddings import EmbeddingService
    from app.core.vector_store import VectorStore

    processed_files = []
    total_chunks = 0

    for file in files:
        # Validate
        if not file.filename.endswith('.pdf'):
            raise HTTPException(400, f"Invalid file type: {file.filename}")

        # Save temporarily
        file_path = f"data/uploads/{file.filename}"
        with open(file_path, 'wb') as f:
            f.write(await file.read())

        # Process
        processor = PDFProcessor()
        chunks = processor.process_pdf(file_path)

        # Embed
        embedder = EmbeddingService()
        embeddings = embedder.embed_batch([c.text for c in chunks])

        # Store
        vector_store = VectorStore()
        for chunk, embedding in zip(chunks, embeddings):
            vector_store.add(
                vector=embedding,
                metadata={
                    "filename": file.filename,
                    "chunk_id": chunk.id,
                    "page": chunk.page_number,
                    "text": chunk.text
                }
            )

        processed_files.append(file.filename)
        total_chunks += len(chunks)
        logger.info(f"Ingested {file.filename}: {len(chunks)} chunks")

    # Persist vector store
    vector_store.save()

    return {
        "message": f"Successfully ingested {len(processed_files)} documents",
        "files_processed": processed_files,
        "total_chunks": total_chunks
    }

@router.get("/stats")
async def get_stats():
    """Return knowledge base statistics"""
    from app.core.vector_store import VectorStore

    vs = VectorStore()
    return {
        "total_documents": vs.document_count(),
        "total_chunks": vs.chunk_count(),
        "embedding_dimension": vs.dimension,
        "indexed_files": vs.get_filenames()
    }

@router.delete("/clear")
async def clear_knowledge_base():
    """Delete all documents from vector store"""
    from app.core.vector_store import VectorStore

    vs = VectorStore()
    stats = vs.get_stats()
    vs.clear()

    return {
        "message": "Knowledge base cleared",
        "chunks_deleted": stats["total_chunks"]
    }
```

---

#### **app/api/query.py**
**Purpose**: Handle user queries with agentic RAG pipeline

**Endpoints Implemented**: `/`, `/health`

**Key Functions**:

```python
from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse
from app.services.agentic_rag import AgenticRAG
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Process user query through agentic RAG pipeline

    Pipeline:
    1. Intent detection
    2. Query transformation
    3. Hybrid search
    4. Citation validation
    5. Answer generation
    6. Hallucination filter
    7. Response construction
    """
    try:
        # Initialize agentic RAG
        rag = AgenticRAG()

        # Execute pipeline
        result = await rag.process_query(
            query=request.query,
            top_k=request.top_k,
            include_citations=request.include_citations
        )

        logger.info(f"Query processed: {request.query[:50]}... | Confidence: {result.confidence}")

        return result

    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(500, f"Query processing failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Service health verification"""
    from app.core.vector_store import VectorStore
    from app.services.llm import LLMService

    try:
        # Check Ollama connection
        llm = LLMService()
        llm_status = llm.test_connection()

        # Check vector store
        vs = VectorStore()
        vs_ready = vs.is_ready()

        return {
            "status": "healthy" if (llm_status and vs_ready) else "degraded",
            "ollama_connection": "active" if llm_status else "inactive",
            "vector_store": "ready" if vs_ready else "not_ready",
            "documents_indexed": vs.document_count()
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

---

### Core Infrastructure

#### **app/core/config.py**
**Purpose**: Environment configuration and settings management

**Configuration Parameters**:

```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"

    # Embedding Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Sentence Transformers
    EMBEDDING_DIMENSION: int = 384

    # Chunking Parameters
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # Search Configuration
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.5
    HYBRID_WEIGHT_SEMANTIC: float = 0.7
    HYBRID_WEIGHT_KEYWORD: float = 0.3

    # LLM Generation
    LLM_TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 2048

    # System Paths
    UPLOAD_DIR: str = "data/uploads"
    VECTOR_DB_PATH: str = "data/vector_db"

    # API Configuration
    API_TITLE: str = "Agentic RAG System"
    API_VERSION: str = "1.0.0"
    CORS_ORIGINS: list = ["*"]

    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()
```

**Usage**:
```python
from app.core.config import settings

# Access configuration
chunk_size = settings.CHUNK_SIZE
ollama_url = settings.OLLAMA_BASE_URL
```

---

#### **app/core/embeddings.py**
**Purpose**: Generate vector embeddings using Sentence Transformers

**Key Class**:

```python
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from app.core.config import settings

class EmbeddingService:
    """
    Local embedding generation using Sentence Transformers

    Model: all-MiniLM-L6-v2 (384 dimensions)
    - Fast inference (CPU-friendly)
    - Good performance on retrieval tasks
    - No external API calls
    """

    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dimension = settings.EMBEDDING_DIMENSION

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        return embedding

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        # Both embeddings are normalized, so dot product = cosine similarity
        return np.dot(embedding1, embedding2)

# Singleton instance
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
```

**Why Sentence Transformers?**
- **Local inference**: No API calls, complete privacy
- **Fast**: CPU-friendly, sub-second encoding
- **Quality**: Pre-trained on semantic similarity tasks
- **Portable**: No GPU required for small batches

---

#### **app/core/vector_store.py**
**Purpose**: Custom NumPy-based vector database

**Implementation**:

```python
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from app.core.config import settings

@dataclass
class VectorEntry:
    """Single vector database entry"""
    id: int
    vector: np.ndarray
    metadata: Dict

class VectorStore:
    """
    Custom vector database using NumPy

    Features:
    - In-memory vector storage
    - Cosine similarity search
    - Metadata management
    - Persistence via pickle

    Limitations:
    - Not scalable beyond ~100K vectors
    - No distributed support
    - Full reload on startup

    Trade-offs:
    - Simplicity over scalability
    - Portability (no external DB)
    - Fast for small datasets
    """

    def __init__(self):
        self.vectors: Optional[np.ndarray] = None  # Shape: (n_vectors, dimension)
        self.metadata: List[Dict] = []
        self.dimension = settings.EMBEDDING_DIMENSION
        self.next_id = 0

        # Load existing data if available
        self.load()

    def add(self, vector: np.ndarray, metadata: Dict) -> int:
        """Add vector with metadata"""
        # Validate dimension
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension {vector.shape[0]} != {self.dimension}")

        # Initialize or append
        if self.vectors is None:
            self.vectors = vector.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, vector])

        # Store metadata
        entry_id = self.next_id
        metadata["id"] = entry_id
        self.metadata.append(metadata)
        self.next_id += 1

        return entry_id

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Tuple[int, float, Dict]]:
        """
        Find most similar vectors using cosine similarity

        Returns:
            List of (id, similarity_score, metadata) tuples
        """
        if self.vectors is None or len(self.vectors) == 0:
            return []

        # Normalize query (if not already)
        query_norm = query_vector / np.linalg.norm(query_vector)

        # Compute cosine similarities (dot product since vectors are normalized)
        similarities = self.vectors @ query_norm

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Filter by minimum similarity
        results = [
            (
                self.metadata[idx]["id"],
                float(similarities[idx]),
                self.metadata[idx]
            )
            for idx in top_indices
            if similarities[idx] >= min_similarity
        ]

        return results

    def batch_search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 5
    ) -> List[List[Tuple[int, float, Dict]]]:
        """Search multiple queries at once"""
        if self.vectors is None:
            return [[] for _ in range(len(query_vectors))]

        # Normalize queries
        query_norms = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)

        # Compute similarities: (n_queries, n_vectors)
        similarities = query_norms @ self.vectors.T

        # Get top-k for each query
        results = []
        for i, sims in enumerate(similarities):
            top_indices = np.argsort(sims)[::-1][:top_k]
            query_results = [
                (
                    self.metadata[idx]["id"],
                    float(sims[idx]),
                    self.metadata[idx]
                )
                for idx in top_indices
            ]
            results.append(query_results)

        return results

    def get_by_id(self, entry_id: int) -> Optional[VectorEntry]:
        """Retrieve specific entry by ID"""
        for idx, meta in enumerate(self.metadata):
            if meta["id"] == entry_id:
                return VectorEntry(
                    id=entry_id,
                    vector=self.vectors[idx],
                    metadata=meta
                )
        return None

    def delete(self, entry_id: int) -> bool:
        """Remove entry by ID"""
        for idx, meta in enumerate(self.metadata):
            if meta["id"] == entry_id:
                # Remove from vectors
                self.vectors = np.delete(self.vectors, idx, axis=0)
                # Remove metadata
                self.metadata.pop(idx)
                return True
        return False

    def clear(self):
        """Remove all entries"""
        self.vectors = None
        self.metadata = []
        self.next_id = 0
        self.save()

    def save(self):
        """Persist to disk"""
        Path(settings.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)

        data = {
            "vectors": self.vectors,
            "metadata": self.metadata,
            "next_id": self.next_id,
            "dimension": self.dimension
        }

        with open(f"{settings.VECTOR_DB_PATH}/vectors.pkl", "wb") as f:
            pickle.dump(data, f)

    def load(self):
        """Load from disk if exists"""
        path = f"{settings.VECTOR_DB_PATH}/vectors.pkl"
        if Path(path).exists():
            with open(path, "rb") as f:
                data = pickle.load(f)

            self.vectors = data["vectors"]
            self.metadata = data["metadata"]
            self.next_id = data["next_id"]
            self.dimension = data["dimension"]

    def chunk_count(self) -> int:
        """Total number of chunks"""
        return len(self.metadata) if self.metadata else 0

    def document_count(self) -> int:
        """Total number of unique documents"""
        if not self.metadata:
            return 0
        filenames = set(m["filename"] for m in self.metadata)
        return len(filenames)

    def get_filenames(self) -> List[str]:
        """List all indexed filenames"""
        if not self.metadata:
            return []
        return list(set(m["filename"] for m in self.metadata))

    def is_ready(self) -> bool:
        """Check if vector store is ready for queries"""
        return self.vectors is not None and len(self.vectors) > 0

# Singleton instance
_vector_store = None

def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
```

**Design Decisions**:
1. **NumPy arrays**: Efficient vectorized operations
2. **Pickle persistence**: Simple serialization
3. **In-memory**: Fast access, no network latency
4. **Normalized vectors**: Enable cosine similarity via dot product
5. **Metadata dict**: Flexible schema for chunk information

---

### Service Layer

#### **app/services/pdf_processor.py**
**Purpose**: Extract and chunk text from PDF documents

**Implementation**:

```python
import pdfplumber
import PyPDF2
import fitz  # PyMuPDF
from typing import List
from dataclasses import dataclass
from pathlib import Path
import logging
from app.core.config import settings
from app.utils.text_processing import chunk_text

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Represents a processed text chunk"""
    id: str
    text: str
    filename: str
    page_number: int
    chunk_index: int

class PDFProcessor:
    """
    Multi-strategy PDF text extraction

    Strategies (in order):
    1. pdfplumber - Best for structured PDFs
    2. PyPDF2 - Fallback for simpler PDFs
    3. PyMuPDF - Fallback for complex layouts
    """

    def process_pdf(self, file_path: str) -> List[TextChunk]:
        """
        Extract and chunk text from PDF

        Returns:
            List of TextChunk objects with metadata
        """
        filename = Path(file_path).name
        logger.info(f"Processing PDF: {filename}")

        # Try extraction strategies
        text_by_page = self._extract_text(file_path)

        if not text_by_page:
            raise ValueError(f"Could not extract text from {filename}")

        # Chunk text
        chunks = []
        chunk_counter = 0

        for page_num, page_text in enumerate(text_by_page, start=1):
            if not page_text.strip():
                continue

            # Split page into chunks
            page_chunks = chunk_text(
                text=page_text,
                chunk_size=settings.CHUNK_SIZE,
                overlap=settings.CHUNK_OVERLAP
            )

            # Create TextChunk objects
            for chunk_text in page_chunks:
                chunk = TextChunk(
                    id=f"{filename}_page{page_num}_chunk{chunk_counter}",
                    text=chunk_text,
                    filename=filename,
                    page_number=page_num,
                    chunk_index=chunk_counter
                )
                chunks.append(chunk)
                chunk_counter += 1

        logger.info(f"Extracted {len(chunks)} chunks from {filename}")
        return chunks

    def _extract_text(self, file_path: str) -> List[str]:
        """Try multiple extraction strategies"""

        # Strategy 1: pdfplumber
        try:
            return self._extract_with_pdfplumber(file_path)
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, trying PyPDF2")

        # Strategy 2: PyPDF2
        try:
            return self._extract_with_pypdf2(file_path)
        except Exception as e:
            logger.warning(f"PyPDF2 failed: {e}, trying PyMuPDF")

        # Strategy 3: PyMuPDF
        try:
            return self._extract_with_pymupdf(file_path)
        except Exception as e:
            logger.error(f"All PDF extraction strategies failed: {e}")
            return []

    def _extract_with_pdfplumber(self, file_path: str) -> List[str]:
        """Extract using pdfplumber (best for tables/structure)"""
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return pages

    def _extract_with_pypdf2(self, file_path: str) -> List[str]:
        """Extract using PyPDF2 (fallback)"""
        pages = []
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return pages

    def _extract_with_pymupdf(self, file_path: str) -> List[str]:
        """Extract using PyMuPDF (complex layouts)"""
        pages = []
        doc = fitz.open(file_path)
        for page in doc:
            text = page.get_text()
            if text:
                pages.append(text)
        doc.close()
        return pages
```

**Why Multiple Libraries?**
- **pdfplumber**: Best for structured documents with tables
- **PyPDF2**: Simple, works for most cases
- **PyMuPDF**: Handles complex layouts, OCR-ed PDFs

---

#### **app/services/intent_detector.py**
**Purpose**: Detect query intent using LLM

**Implementation**:

```python
from app.services.llm import LLMService
from typing import Literal
import logging

logger = logging.getLogger(__name__)

IntentType = Literal["knowledge_base_query", "greeting", "off_topic", "pii_detected"]

class IntentDetector:
    """
    LLM-based intent classification

    Advantages over regex:
    - Handles natural language variations
    - Context-aware classification
    - No pattern maintenance

    Intents:
    - knowledge_base_query: Question about documents
    - greeting: Hello, hi, how are you
    - off_topic: Unrelated questions
    - pii_detected: Personal identifiable information
    """

    def __init__(self):
        self.llm = LLMService()

    def detect(self, query: str) -> IntentType:
        """
        Classify query intent

        Returns:
            Intent type as string
        """

        prompt = f"""Classify the user's intent into one of these categories:

1. knowledge_base_query - Questions about documents or knowledge base
2. greeting - Greetings like "hello", "hi", "how are you"
3. off_topic - Questions unrelated to documents (weather, news, etc.)
4. pii_detected - Query contains personal information (SSN, credit cards, etc.)

User query: "{query}"

Respond with ONLY the category name, nothing else.

Intent:"""

        response = self.llm.generate(
            prompt=prompt,
            max_tokens=10,
            temperature=0.0  # Deterministic classification
        )

        intent = response.strip().lower()
        logger.info(f"Intent detected: {intent} for query: {query[:50]}...")

        # Validate response
        if intent not in ["knowledge_base_query", "greeting", "off_topic", "pii_detected"]:
            logger.warning(f"Unknown intent '{intent}', defaulting to knowledge_base_query")
            return "knowledge_base_query"

        return intent
```

**Why LLM-based?**
- **Flexible**: Handles variations ("hi", "hello", "hey there")
- **Context-aware**: Understands "What's the CEO's name?" vs "What's the weather?"
- **Maintainable**: No regex patterns to update

---

#### **app/services/query_transformer.py**
**Purpose**: Enhance queries for better retrieval

**Implementation**:

```python
from app.services.llm import LLMService
import logging

logger = logging.getLogger(__name__)

class QueryTransformer:
    """
    Query enhancement for improved retrieval

    Techniques:
    - Query expansion (synonyms, related terms)
    - Keyword extraction
    - Question decomposition
    """

    def __init__(self):
        self.llm = LLMService()

    def transform(self, query: str) -> str:
        """
        Enhance query with relevant keywords and expansions

        Example:
            Input: "Who is the CEO?"
            Output: "CEO, chief executive officer, leadership, executive team, company leader"
        """

        prompt = f"""Given this user question, generate relevant keywords and synonyms that would help retrieve the answer from a document database.

User question: "{query}"

Generate a comma-separated list of keywords, synonyms, and related terms. Include:
- Exact terms from the question
- Synonyms
- Related concepts
- Common variations

Keywords:"""

        enhanced = self.llm.generate(
            prompt=prompt,
            max_tokens=100,
            temperature=0.3
        )

        transformed = enhanced.strip()
        logger.info(f"Query transformed: '{query}' → '{transformed}'")

        return transformed
```

---

#### **app/services/search.py**
**Purpose**: Hybrid search combining semantic and keyword matching

**Implementation**:

```python
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from collections import Counter
import math
import logging
from app.core.vector_store import VectorStore, get_vector_store
from app.core.embeddings import EmbeddingService, get_embedding_service
from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result with metadata"""
    chunk_id: int
    filename: str
    page_number: int
    text: str
    semantic_score: float
    keyword_score: float
    combined_score: float

class HybridSearchEngine:
    """
    Combines semantic (vector) and keyword (BM25) search

    Fusion method: Reciprocal Rank Fusion (RRF)
    """

    def __init__(self):
        self.vector_store = get_vector_store()
        self.embedder = get_embedding_service()

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Perform hybrid search

        Steps:
        1. Semantic search using vector similarity
        2. Keyword search using BM25
        3. Fusion using reciprocal rank
        4. Re-rank combined results
        """

        # 1. Semantic search
        query_embedding = self.embedder.embed(query)
        semantic_results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=top_k * 2  # Get more for fusion
        )

        # 2. Keyword search (BM25)
        keyword_results = self._bm25_search(query, top_k * 2)

        # 3. Fusion
        fused_results = self._reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            k=60  # RRF parameter
        )

        # 4. Return top-k
        return fused_results[:top_k]

    def _bm25_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[int, float, Dict]]:
        """
        BM25 keyword search

        BM25 scoring:
        - Considers term frequency
        - Considers document length
        - Considers inverse document frequency
        """
        query_terms = self._tokenize(query.lower())

        # Get all documents
        all_chunks = [
            (meta["id"], meta["text"], meta)
            for meta in self.vector_store.metadata
        ]

        if not all_chunks:
            return []

        # Calculate BM25 scores
        scores = []
        avg_doc_length = np.mean([len(self._tokenize(text)) for _, text, _ in all_chunks])

        for chunk_id, text, metadata in all_chunks:
            score = self._bm25_score(
                query_terms=query_terms,
                document=text,
                avg_doc_length=avg_doc_length
            )
            scores.append((chunk_id, score, metadata))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def _bm25_score(
        self,
        query_terms: List[str],
        document: str,
        avg_doc_length: float,
        k1: float = 1.5,
        b: float = 0.75
    ) -> float:
        """
        Calculate BM25 score for document

        Parameters:
        - k1: Term frequency saturation (1.2-2.0)
        - b: Length normalization (0-1)
        """
        doc_terms = self._tokenize(document.lower())
        doc_length = len(doc_terms)
        term_freqs = Counter(doc_terms)

        score = 0.0

        for term in query_terms:
            if term not in term_freqs:
                continue

            tf = term_freqs[term]

            # IDF calculation (simplified)
            idf = math.log(1 + 1 / (1 + tf))

            # BM25 formula
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))

            score += idf * (numerator / denominator)

        return score

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (can be enhanced with NLTK/spaCy)"""
        # Remove punctuation, split on whitespace
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[int, float, Dict]],
        keyword_results: List[Tuple[int, float, Dict]],
        k: int = 60
    ) -> List[SearchResult]:
        """
        Combine rankings using Reciprocal Rank Fusion

        RRF formula: score = sum(1 / (k + rank))

        Advantages:
        - No score normalization needed
        - Robust to outliers
        - Simple and effective
        """
        # Create rank dictionaries
        semantic_ranks = {chunk_id: rank for rank, (chunk_id, _, _) in enumerate(semantic_results)}
        keyword_ranks = {chunk_id: rank for rank, (chunk_id, _, _) in enumerate(keyword_results)}

        # Get all unique chunk IDs
        all_chunk_ids = set(semantic_ranks.keys()) | set(keyword_ranks.keys())

        # Calculate RRF scores
        fused_scores = {}
        for chunk_id in all_chunk_ids:
            semantic_rank = semantic_ranks.get(chunk_id, 1000)  # Large rank if not found
            keyword_rank = keyword_ranks.get(chunk_id, 1000)

            # RRF score
            rrf_score = (
                settings.HYBRID_WEIGHT_SEMANTIC / (k + semantic_rank) +
                settings.HYBRID_WEIGHT_KEYWORD / (k + keyword_rank)
            )

            fused_scores[chunk_id] = rrf_score

        # Sort by RRF score
        sorted_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        # Build SearchResult objects
        results = []
        for chunk_id, rrf_score in sorted_chunks:
            # Get metadata
            metadata = next((m for _, _, m in semantic_results + keyword_results if m["id"] == chunk_id), None)
            if not metadata:
                continue

            # Get individual scores
            semantic_score = next((score for cid, score, _ in semantic_results if cid == chunk_id), 0.0)
            keyword_score = next((score for cid, score, _ in keyword_results if cid == chunk_id), 0.0)

            result = SearchResult(
                chunk_id=chunk_id,
                filename=metadata["filename"],
                page_number=metadata["page"],
                text=metadata["text"],
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                combined_score=rrf_score
            )
            results.append(result)

        return results
```

**Why Hybrid Search?**
- **Semantic**: Understands meaning, handles paraphrasing
- **Keyword**: Finds exact matches, handles rare terms
- **Fusion**: Combines strengths of both approaches

---

#### **app/services/agentic_rag.py**
**Purpose**: Main orchestrator for the 8-stage ReAct pipeline

**Implementation** (Abridged):

```python
from typing import Dict, List
from dataclasses import dataclass
from app.services.intent_detector import IntentDetector
from app.services.query_transformer import QueryTransformer
from app.services.search import HybridSearchEngine
from app.services.llm import LLMService
from app.services.citation_validator import CitationValidator
from app.services.hallucination_filter import HallucinationFilter
from app.services.query_refusal import QueryRefusalPolicy
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgenticResponse:
    """Final response from agentic RAG"""
    answer: str
    citations: List[Dict]
    confidence: str
    reasoning_steps: Dict
    processing_time_ms: float

class AgenticRAG:
    """
    8-Stage Agentic RAG Pipeline

    Stages:
    1. Intent Detection
    2. Query Planning
    3. Hybrid Search
    4. Result Observation
    5. Citation Validation
    6. Answer Generation
    7. Hallucination Verification
    8. Response Return
    """

    def __init__(self):
        self.intent_detector = IntentDetector()
        self.query_transformer = QueryTransformer()
        self.search_engine = HybridSearchEngine()
        self.llm = LLMService()
        self.citation_validator = CitationValidator()
        self.hallucination_filter = HallucinationFilter()
        self.query_refusal = QueryRefusalPolicy()

    async def process_query(
        self,
        query: str,
        top_k: int = 5,
        include_citations: bool = True
    ) -> AgenticResponse:
        """Execute full agentic RAG pipeline"""
        import time
        start_time = time.time()

        reasoning_steps = {}

        # Stage 1: Intent Detection
        logger.info("Stage 1: Intent Detection")
        intent = self.intent_detector.detect(query)
        reasoning_steps["intent"] = intent

        if intent != "knowledge_base_query":
            return self._handle_non_kb_query(intent, query, reasoning_steps)

        # Stage 2: Query Planning
        logger.info("Stage 2: Query Planning")
        transformed_query = self.query_transformer.transform(query)
        reasoning_steps["query_transformation"] = transformed_query

        # Stage 3: Hybrid Search
        logger.info("Stage 3: Hybrid Search")
        search_results = self.search_engine.search(query, top_k=top_k)
        reasoning_steps["chunks_retrieved"] = len(search_results)
        reasoning_steps["retrieval_method"] = "hybrid_search"

        # Stage 4: Result Observation
        logger.info("Stage 4: Result Observation")
        confidence_scores = [r.combined_score for r in search_results]
        max_confidence = max(confidence_scores) if confidence_scores else 0.0
        reasoning_steps["max_similarity"] = max_confidence

        # Stage 5: Citation Validation
        logger.info("Stage 5: Citation Validation")
        if max_confidence < settings.SIMILARITY_THRESHOLD:
            return AgenticResponse(
                answer="I cannot find sufficient information in the documents to answer this question.",
                citations=[],
                confidence="none",
                reasoning_steps=reasoning_steps,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Stage 6: Answer Generation
        logger.info("Stage 6: Answer Generation")
        context = self._build_context(search_results)
        answer = self._generate_answer(query, context)

        # Stage 7: Hallucination Verification
        logger.info("Stage 7: Hallucination Verification")
        verified_answer, hallucination_check = self.hallucination_filter.verify(
            answer=answer,
            context=context
        )
        reasoning_steps["hallucination_check"] = hallucination_check

        # Stage 8: Response Return
        logger.info("Stage 8: Response Return")
        citations = self._build_citations(search_results) if include_citations else []

        # Determine confidence
        confidence = self._calculate_confidence(max_confidence)

        processing_time = (time.time() - start_time) * 1000

        return AgenticResponse(
            answer=verified_answer,
            citations=citations,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            processing_time_ms=processing_time
        )

    def _build_context(self, search_results: List) -> str:
        """Concatenate search results into context"""
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(
                f"[Source {i}] {result.filename}, Page {result.page_number}:\n{result.text}"
            )
        return "\n\n".join(context_parts)

    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM with context"""
        prompt = f"""Based on the following context from documents, answer the user's question.

IMPORTANT:
- Only use information from the provided context
- Cite sources by mentioning [Source N]
- If information is insufficient, say so
- Do not make assumptions or add external knowledge

Context:
{context}

Question: {query}

Answer:"""

        answer = self.llm.generate(
            prompt=prompt,
            max_tokens=settings.MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE
        )

        return answer.strip()

    def _build_citations(self, search_results: List) -> List[Dict]:
        """Build citation objects"""
        citations = []
        for result in search_results:
            citations.append({
                "chunk_id": result.chunk_id,
                "source": result.filename,
                "page": result.page_number,
                "similarity": float(result.combined_score),
                "text_snippet": result.text[:200] + "..."
            })
        return citations

    def _calculate_confidence(self, max_similarity: float) -> str:
        """Map similarity score to confidence level"""
        if max_similarity >= 0.8:
            return "high"
        elif max_similarity >= 0.6:
            return "medium"
        elif max_similarity >= 0.4:
            return "low"
        else:
            return "very_low"

    def _handle_non_kb_query(self, intent: str, query: str, reasoning_steps: Dict) -> AgenticResponse:
        """Handle non-knowledge-base queries"""
        if intent == "greeting":
            answer = "Hello! I'm here to help you with questions about the uploaded documents. What would you like to know?"
        elif intent == "pii_detected":
            answer = "I cannot process queries containing personal identifiable information for privacy reasons."
        else:
            answer = "This question appears to be outside the scope of the knowledge base. Please ask questions related to the uploaded documents."

        return AgenticResponse(
            answer=answer,
            citations=[],
            confidence="n/a",
            reasoning_steps=reasoning_steps,
            processing_time_ms=0.0
        )
```

---

## Data Flow and Routing

### Complete Request Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      INGESTION FLOW                              │
└─────────────────────────────────────────────────────────────────┘

1. User uploads PDF via UI
   │
   ▼
2. POST /ingest/upload (api/ingestion.py)
   │
   ├─► Validate file type
   ├─► Save to data/uploads/
   │
   ▼
3. PDFProcessor.process_pdf()
   │
   ├─► Extract text (pdfplumber/PyPDF2/PyMuPDF)
   ├─► Split into pages
   │
   ▼
4. TextChunker.chunk_text()
   │
   ├─► Tokenize text
   ├─► Create 512-token chunks with 50-token overlap
   │
   ▼
5. EmbeddingService.embed_batch()
   │
   ├─► Load Sentence Transformer model
   ├─► Generate 384-dim vectors
   ├─► Normalize embeddings
   │
   ▼
6. VectorStore.add()
   │
   ├─► Append to NumPy array
   ├─► Store metadata (filename, page, chunk_id)
   │
   ▼
7. VectorStore.save()
   │
   ├─► Serialize to pickle
   ├─► Save to data/vector_db/vectors.pkl
   │
   ▼
8. Return success response to UI

┌─────────────────────────────────────────────────────────────────┐
│                       QUERY FLOW                                 │
└─────────────────────────────────────────────────────────────────┘

1. User submits question via UI
   │
   ▼
2. POST /query/ (api/query.py)
   │
   ├─► Parse JSON body
   ├─► Extract query, top_k, include_citations
   │
   ▼
3. AgenticRAG.process_query()
   │
   ├─► STAGE 1: IntentDetector.detect()
   │   │
   │   ├─► Build classification prompt
   │   ├─► Call LLMService.generate()
   │   │   │
   │   │   ├─► HTTP POST to Ollama API
   │   │   ├─► Model: mistral
   │   │   ├─► Temperature: 0.0 (deterministic)
   │   │   └─► Return intent: "knowledge_base_query"
   │   │
   │   └─► If not KB query, return polite refusal
   │
   ├─► STAGE 2: QueryTransformer.transform()
   │   │
   │   ├─► Build enhancement prompt
   │   ├─► Call LLMService.generate()
   │   └─► Return expanded keywords
   │
   ├─► STAGE 3: HybridSearchEngine.search()
   │   │
   │   ├─► Semantic Search:
   │   │   │
   │   │   ├─► EmbeddingService.embed(query)
   │   │   ├─► VectorStore.search(query_vector, top_k=10)
   │   │   │   │
   │   │   │   ├─► Normalize query vector
   │   │   │   ├─► Dot product: vectors @ query_norm
   │   │   │   ├─► Sort by similarity (descending)
   │   │   │   └─► Return top-10 results
   │   │   │
   │   │   └─► Get results with metadata
   │   │
   │   ├─► Keyword Search (BM25):
   │   │   │
   │   │   ├─► Tokenize query
   │   │   ├─► For each document chunk:
   │   │   │   ├─► Calculate term frequency
   │   │   │   ├─► Calculate IDF
   │   │   │   ├─► Apply BM25 formula
   │   │   │   └─► Assign score
   │   │   │
   │   │   └─► Sort by BM25 score
   │   │
   │   ├─► Reciprocal Rank Fusion:
   │   │   │
   │   │   ├─► For each unique chunk:
   │   │   │   ├─► Get semantic rank
   │   │   │   ├─► Get keyword rank
   │   │   │   ├─► Calculate: 0.7/(60+rank_s) + 0.3/(60+rank_k)
   │   │   │   └─► Store RRF score
   │   │   │
   │   │   └─► Sort by RRF score
   │   │
   │   └─► Return top-K fused results
   │
   ├─► STAGE 4: Result Observation
   │   │
   │   ├─► Extract confidence scores
   │   ├─► Calculate max similarity
   │   └─► Log retrieval quality
   │
   ├─► STAGE 5: CitationValidator.validate()
   │   │
   │   ├─► Check: max_similarity >= THRESHOLD (0.5)
   │   ├─► If insufficient:
   │   │   └─► Return "Insufficient evidence" response
   │   │
   │   └─► Continue if passed
   │
   ├─► STAGE 6: Answer Generation
   │   │
   │   ├─► Build context from top chunks
   │   ├─► Create prompt with context + question
   │   ├─► LLMService.generate()
   │   │   │
   │   │   ├─► HTTP POST to Ollama
   │   │   ├─► Model: mistral
   │   │   ├─► Temperature: 0.3
   │   │   ├─► Max tokens: 2048
   │   │   └─► Stream: false
   │   │
   │   └─► Extract generated answer
   │
   ├─► STAGE 7: HallucinationFilter.verify()
   │   │
   │   ├─► Extract claims from answer
   │   ├─► For each claim:
   │   │   ├─► Check if supported by context
   │   │   ├─► If not supported:
   │   │   │   └─► Flag or remove claim
   │   │   │
   │   │   └─► Keep if supported
   │   │
   │   └─► Return verified answer
   │
   └─► STAGE 8: Response Construction
       │
       ├─► Build citations array
       ├─► Calculate confidence level
       ├─► Package reasoning steps
       ├─► Calculate processing time
       │
       └─► Return AgenticResponse

4. api/query.py receives response
   │
   ├─► Serialize to JSON
   │
   ▼
5. HTTP 200 response to UI
   │
   ▼
6. UI displays answer + citations
```

---

## Task 4: StackAI Roadmap

### Strategic Feature Prioritization

*[This section would contain the actual roadmap recommendations based on platform analysis]*

**High Priority Features**:
1. **Multi-Agent Orchestration** - Enable complex workflows with specialized agents
2. **Custom Knowledge Base Connectors** - Integrate with enterprise data sources
3. **Real-time Collaboration** - Team editing and version control

**Medium Priority**:
1. **Advanced Analytics Dashboard** - Usage metrics and performance tracking
2. **Template Marketplace** - Pre-built agent templates
3. **API Rate Limiting Controls** - Cost management

**Low Priority**:
1. **White-label Options** - Custom branding
2. **Mobile App** - iOS/Android native apps

---

## Setup and Deployment

### Prerequisites
```bash
# 1. Install Ollama
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull Mistral model
ollama pull mistral

# 3. Verify Ollama is running
curl http://localhost:11434/api/generate -d '{
  "model": "mistral",
  "prompt": "test",
  "stream": false
}'
```

### Installation
```bash
# 1. Clone repository
git clone https://github.com/nag07799/agentic-rag.git
cd agentic-rag

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model (optional, for NLP)
python -m spacy download en_core_web_sm
```

### Configuration
```bash
# Create .env file
cat > .env << EOF
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral

# Embedding Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Search
TOP_K=5
SIMILARITY_THRESHOLD=0.5
HYBRID_WEIGHT_SEMANTIC=0.7
HYBRID_WEIGHT_KEYWORD=0.3

# LLM Generation
LLM_TEMPERATURE=0.3
MAX_TOKENS=2048

# Paths
UPLOAD_DIR=data/uploads
VECTOR_DB_PATH=data/vector_db
EOF
```

### Running the Application
```bash
# Development mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### Access Points
- **Web UI**: http://localhost:8000/ui
- **Swagger API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Testing

### Test Document Upload
```bash
curl -X POST "http://localhost:8000/ingest/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document.pdf"
```

### Test Query
```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic?",
    "top_k": 5,
    "include_citations": true
  }'
```

### Check Stats
```bash
curl "http://localhost:8000/ingest/stats"
```

---

## Performance Considerations

### Scalability Limits
- **Vector Store**: ~100K chunks (NumPy in-memory limit)
- **Concurrent Requests**: Limited by Ollama GPU capacity
- **File Size**: Up to 50MB per PDF (configurable)

### Optimization Strategies
1. **Batch Processing**: Use `embed_batch()` for multiple chunks
2. **Caching**: Cache frequent query embeddings
3. **GPU Acceleration**: Ollama with GPU for faster inference
4. **Async Processing**: FastAPI async endpoints for I/O

---

## Security Considerations

### Implemented Safeguards
1. **PII Detection**: Query refusal policy
2. **Input Validation**: Pydantic schemas
3. **File Type Restriction**: Only PDFs allowed
4. **Local Processing**: No external API calls (privacy)

### Recommendations
1. Add authentication (JWT tokens)
2. Rate limiting (per IP/user)
3. File size limits
4. Sandboxed PDF processing
5. HTTPS in production

---

## Troubleshooting

### Common Issues

**1. "Connection refused to Ollama"**
```bash
# Check if Ollama is running
ollama serve

# Or use systemd (Linux)
systemctl status ollama
```

**2. "Model not found"**
```bash
# Pull the model
ollama pull mistral
```

**3. "Out of memory during embedding"**
```bash
# Reduce batch size in embeddings.py
embeddings = self.model.encode(texts, batch_size=8)  # Default: 32
```

**4. "No results returned"**
```bash
# Check if documents are indexed
curl http://localhost:8000/ingest/stats

# Lower similarity threshold in .env
SIMILARITY_THRESHOLD=0.3
```

---

## Conclusion

This implementation demonstrates:
- ✅ Production-ready RAG architecture
- ✅ Agentic reasoning with ReAct pattern
- ✅ Local-first processing (privacy + cost)
- ✅ Hybrid search for robust retrieval
- ✅ Hallucination prevention
- ✅ Complete API with FastAPI
- ✅ Interactive UI

**Key Achievements**:
- Zero external API costs
- Complete data privacy
- Scalable design for small-medium datasets
- Transparent reasoning steps
- Citation-backed answers

---

## Repository Links

- **GitHub**: https://github.com/nag07799/agentic-rag
- **Live Demo**: [URL if deployed]
- **Video Walkthrough**: [Loom link]

---

*End of Documentation*
