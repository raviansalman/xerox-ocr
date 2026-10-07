# Vector Storage Processing Service

A comprehensive document processing pipeline with PII protection, text chunking, embeddings, and vector database integration. Built with Python, Unstructured, Presidio, LangChain, SentenceTransformers, and Milvus.

## Features

### Core Document Processing
- **Multi-format Document Processing**: PDF, images, audio, video, and more
- **PII Protection**: Automatic detection and redaction of sensitive information using Microsoft Presidio
- **Intelligent Chunking**: Token-aware text splitting with LangChain RecursiveCharacterTextSplitter
- **Vector Embeddings**: High-quality embeddings with SentenceTransformers (CPU-optimized)
- **Vector Database**: Milvus integration for efficient similarity search
- **Background Processing**: Celery workers for scalable document processing
- **Web UI**: Modern interface for document upload and search
- **Docker Support**: Complete containerized deployment

### Advanced OCR Features
- Multiple preprocessing techniques (contrast, sharpening, resizing)
- Fallback OCR configurations for maximum text extraction
- Image enhancement for better OCR accuracy
- Support for scanned documents and image-based PDFs
- Tesseract OCR with multiple PSM modes

### Visual Intelligence
- Object detection using YOLO (when available) or OpenCV
- Image captioning with BLIP model
- Basic computer vision with OpenCV cascades
- Visual element extraction for searchable content

## Prerequisites

### System Requirements
- **Docker and Docker Compose**
- **Python**: 3.12+ (for local development)
- **Tesseract OCR**: For text extraction
- **Memory**: 4GB+ RAM recommended
- **Storage**: 1GB+ for models and dependencies
- **Ports**: 8000, 8080, 19530, 6380 available

### Supported Operating Systems
- Linux (Ubuntu 20.04+, CentOS 7+)
- macOS (10.15+)
- Windows (10+ with WSL2 recommended)

## Quick Start

### Option 1: Standard Pipeline (Recommended)

#### 1. Clone the Repository
```bash
git clone git@github.com:StorageChain-LLC/vector-storage-processing-service-python.git
cd vector-storage-processing-service-python
```

#### 2. Environment Setup
Copy the environment template and configure:
```bash
cp env.example .env
```

Edit `.env` file with your configuration:
```bash
# JWT Secret for authentication
JWT_SECRET_KEY=your-secret-key-here

# Milvus Configuration
MILVUS_HOST=milvus
MILVUS_PORT=19530

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# Optional: External services
OPENAI_API_KEY=your-openai-key
```

#### 3. Start the Services
```bash
docker-compose up -d
```

#### 4. Wait for Container Health Checks
**IMPORTANT**: The system uses health checks to ensure all services are ready. Wait for all containers to be healthy before using the application.

Check container status:
```bash
docker-compose ps
```

All services should show "healthy" status:
- ✅ `api` - Document processing API
- ✅ `milvus` - Vector database
- ✅ `redis` - Cache and session storage
- ✅ `celery-worker` - Background processing

**Expected startup time**: 2-10 minutes for first run (includes model downloads)

#### 5. Access the Application
Once all containers are healthy:
- **Web UI**: http://localhost:8080
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Option 2: Ultimate Search Processor

For advanced OCR and visual intelligence features:

#### 1. Navigate to Ultimate Directory
```bash
cd ultimate
```

#### 2. Start Ultimate Services
```bash
./start.sh
```

This will:
- Build and start all Ultimate services
- Wait for services to initialize
- Check system health
- Display access information

#### 3. Access Ultimate Interface
- **Main Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Project Structure

```
vector-storage-processing-service-python/
├── src/                           # Core source code
│   ├── api.py                     # FastAPI application
│   ├── normalize.py               # Document normalization
│   ├── redaction.py               # PII redaction
│   ├── chunking.py                # Text chunking
│   ├── embeddings.py              # Vector embeddings
│   ├── vector_db_milvus_server.py # Milvus integration
│   ├── tasks.py                   # Celery background tasks
│   └── ...
├── ultimate/                      # Ultimate Search Processor
│   ├── src/                       # Ultimate source code
│   │   ├── ultimate_search_processor.py
│   │   ├── ultimate_vector_integration.py
│   │   ├── ultimate_tasks.py
│   │   └── ...
│   ├── scripts/                   # Utility scripts
│   ├── tests/                     # Test files
│   ├── docs/                      # Documentation
│   ├── ultimate_ui.py            # Ultimate web interface
│   ├── docker-compose.ultimate.yml
│   ├── Dockerfile.ultimate
│   ├── requirements_ultimate.txt
│   └── start.sh                  # Combined build and start script
├── scripts/                       # Utility scripts
├── docs/                          # Documentation
├── data/                          # Processed data storage
├── logs/                          # Application logs
├── temp_uploads/                  # Temporary file storage
├── docker-compose.yml             # Standard Docker services
├── Dockerfile                     # Standard container configuration
└── requirements.txt               # Python dependencies
```

## Usage

### Upload Documents

1. Open the web interface (http://localhost:8080 for standard, http://localhost:8000 for Ultimate)
2. Select documents (PDF, images, audio, video)
3. Documents are automatically processed in the background
4. Monitor progress in the UI

### Search Documents

1. Use the search interface to find relevant content
2. Search supports semantic similarity and keyword matching
3. Results show document metadata and relevant chunks

## API Endpoints

### Standard Pipeline API

The standard pipeline provides a comprehensive REST API with authentication and tenant isolation.

#### Authentication
All endpoints require an API key passed as a query parameter:
```bash
curl "http://localhost:8000/health?api_key=your-secret-api-key-here"
```

#### Health and Status
- **GET `/health`** - System health check
  ```bash
  curl "http://localhost:8000/health?api_key=your-api-key"
  ```

- **GET `/stats`** - Pipeline statistics
  ```bash
  curl "http://localhost:8000/stats?api_key=your-api-key"
  ```

#### Document Processing
- **POST `/documents/process`** - Process document from URL (asynchronous)
  ```bash
  curl -X POST "http://localhost:8000/documents/process?api_key=your-api-key" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "url=https://example.com/document.pdf&object_id=doc123&user_id=user123"
  ```

- **POST `/documents/process-fast`** - Process uploaded file (synchronous)
  ```bash
  curl -X POST "http://localhost:8000/documents/process-fast?api_key=your-api-key" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@document.pdf" \
    -F "object_id=doc123" \
    -F "user_id=user123"
  ```

- **POST `/documents/process-sync`** - Process uploaded file (synchronous, backward compatibility)
  ```bash
  curl -X POST "http://localhost:8000/documents/process-sync?api_key=your-api-key" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@document.pdf" \
    -F "object_id=doc123" \
    -F "user_id=user123"
  ```

#### Search
- **POST `/search`** - Search documents (JSON request)
  ```bash
  curl -X POST "http://localhost:8000/search?api_key=your-api-key" \
    -H "Content-Type: application/json" \
    -d '{
      "query": "contract terms",
      "limit": 10,
      "user_id": "user123",
      "file_types": ["pdf"],
      "score_threshold": 0.3
    }'
  ```

- **GET `/search`** - Search documents (query parameters)
  ```bash
  curl "http://localhost:8000/search?api_key=your-api-key&q=contract%20terms&limit=10&user_id=user123"
  ```

- **GET `/search/fast`** - Fast search using optimized processor
  ```bash
  curl "http://localhost:8000/search/fast?api_key=your-api-key&q=contract%20terms&limit=10&user_id=user123"
  ```

#### Task Management
- **GET `/tasks/{task_id}`** - Get task status
  ```bash
  curl "http://localhost:8000/tasks/task-uuid-here?api_key=your-api-key"
  ```

- **GET `/tasks/{task_id}/result`** - Get task result
  ```bash
  curl "http://localhost:8000/tasks/task-uuid-here/result?api_key=your-api-key"
  ```

- **DELETE `/tasks/{task_id}`** - Cancel task
  ```bash
  curl -X DELETE "http://localhost:8000/tasks/task-uuid-here?api_key=your-api-key"
  ```

- **GET `/progress/{object_id}`** - Get progress by object ID
  ```bash
  curl "http://localhost:8000/progress/doc123?api_key=your-api-key"
  ```

#### Administration
- **POST `/admin/drop-collection`** - Drop Milvus collection
  ```bash
  curl -X POST "http://localhost:8000/admin/drop-collection?api_key=your-api-key"
  ```

### Ultimate Search Processor API

The Ultimate Search Processor provides advanced OCR and visual intelligence capabilities.

#### Health and Status
- **GET `/health`** - System health check with vector database stats
  ```bash
  curl "http://localhost:8000/health"
  ```

#### Document Processing
- **POST `/process-file`** - Process uploaded file with advanced OCR
  ```bash
  curl -X POST "http://localhost:8000/process-file" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@document.pdf"
  ```

- **POST `/process-url`** - Process document from URL
  ```bash
  curl -X POST "http://localhost:8000/process-url" \
    -H "Content-Type: application/json" \
    -d '{"url": "https://example.com/document.pdf"}'
  ```

- **POST `/process`** - Process document for backend integration
  ```bash
  curl -X POST "http://localhost:8000/process" \
    -H "Content-Type: application/json" \
    -d '{
      "fileUrl": "https://example.com/document.pdf",
      "userId": "user123",
      "fileId": "doc123"
    }'
  ```

- **POST `/process-background`** - Process large files in background using Celery
  ```bash
  curl -X POST "http://localhost:8000/process-background" \
    -H "Content-Type: application/json" \
    -d '{
      "fileUrl": "https://example.com/large-document.pdf",
      "userId": "user123",
      "fileId": "doc123"
    }'
  ```

#### Search
- **POST `/search-vector`** - Vector similarity search using Milvus
  ```bash
  curl -X POST "http://localhost:8000/search-vector" \
    -H "Content-Type: application/json" \
    -d '{
      "query": "contract terms",
      "userId": "user123",
      "limit": 5,
      "similarity_threshold": 0.5
    }'
  ```

- **POST `/search-files`** - Search across all processed files
  ```bash
  curl -X POST "http://localhost:8000/search-files" \
    -H "Content-Type: application/json" \
    -d '{"query": "contract terms"}'
  ```

- **POST `/search`** - Search documents for backend integration
  ```bash
  curl -X POST "http://localhost:8000/search" \
    -H "Content-Type: application/json" \
    -d '{
      "searchedText": "contract terms",
      "userId": "user123",
      "limit": 10
    }'
  ```

#### File Management
- **GET `/files`** - Get all processed files
  ```bash
  curl "http://localhost:8000/files"
  ```

- **POST `/delete-file`** - Delete a processed file
  ```bash
  curl -X POST "http://localhost:8000/delete-file" \
    -H "Content-Type: application/json" \
    -d '{"file_id": "file_1234567890_document"}'
  ```

#### Task Management
- **GET `/task-status/{task_id}`** - Get background task status
  ```bash
  curl "http://localhost:8000/task-status/task-uuid-here"
  ```

### Response Formats

#### Standard Pipeline Responses

**Document Processing Response:**
```json
{
  "file_path": "processed/document.pdf",
  "file_name": "document.pdf",
  "file_type": "pdf",
  "file_size": 1024000,
  "total_pages": 10,
  "extraction_method": "pymupdf",
  "pii_entities_found": 5,
  "pii_entities": ["EMAIL", "PHONE"],
  "chunks_created": 25,
  "chunks_stored": 25,
  "processing_time": 2.5,
  "success": true,
  "error": null
}
```

**Search Response:**
```json
{
  "query": "contract terms",
  "total_results": 5,
  "results": [
    {
      "chunk_id": "chunk_123",
      "text": "The contract terms specify...",
      "score": 0.85,
      "metadata": {
        "page_number": 1,
        "object_id": "doc123"
      },
      "object_id": "doc123"
    }
  ],
  "processing_time": 0.15,
  "search_id": "search_uuid"
}
```

#### Ultimate Search Processor Responses

**Health Response:**
```json
{
  "status": "healthy",
  "processor_available": true,
  "vector_integration_available": true,
  "timestamp": "2024-01-15 10:30:00",
  "vector_stats": {
    "total_documents": 50,
    "total_chunks": 1250,
    "collections": ["chunks_bge_small"]
  }
}
```

**Vector Search Response:**
```json
{
  "success": true,
  "query": "contract terms",
  "results": [
    {
      "chunk_id": "chunk_123",
      "text": "The contract terms specify...",
      "score": 0.85,
      "metadata": {
        "page_number": 1,
        "source_file": "document.pdf",
        "element_type": "text"
      }
    }
  ],
  "total_results": 5,
  "processing_time": 0.12
}
```

### Error Handling

All endpoints return appropriate HTTP status codes:
- **200**: Success
- **400**: Bad Request (invalid parameters)
- **401**: Unauthorized (invalid API key)
- **404**: Not Found
- **500**: Internal Server Error

Error responses include details:
```json
{
  "detail": "Invalid API key"
}
```

## Development Setup

### Local Development (without Docker)

1. **Create Virtual Environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies**:
```bash
# For standard pipeline
pip install -r requirements.txt

# For Ultimate Search Processor
pip install -r ultimate/requirements_ultimate.txt
```

3. **Download spaCy Models** (for standard pipeline):
```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

4. **Install System Dependencies** (for Ultimate):
```bash
# Ubuntu/Debian
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract
```

5. **Start External Services**:
```bash
# Start Milvus
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:v2.4.0

# Start Redis
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

6. **Run the Application**:
```bash
# Standard pipeline
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
python serve_ui.py

# Ultimate Search Processor
cd ultimate
python ultimate_ui.py
```

## Configuration

### Environment Variables

#### Standard Pipeline
```bash
# JWT Secret for authentication
JWT_SECRET_KEY=your-secret-key-here

# Milvus Configuration
MILVUS_HOST=milvus
MILVUS_PORT=19530

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# Optional: External services
OPENAI_API_KEY=your-openai-key
```

#### Ultimate Search Processor
```bash
# Tesseract configuration
export TESSERACT_CMD=/usr/bin/tesseract

# Python path
export PYTHONPATH=/app

# Optional: Custom OCR language
export OCR_LANGUAGE=eng

# Workflow Stage Integration (Optional)
export WORKFLOW_API_URL=http://localhost:3333
export WORKFLOW_ACCESS_KEY=your_access_key
export WORKFLOW_STOR_API_KEY=your_stor_api_key
export WORKFLOW_ENABLED=true
```

### Workflow Stage Integration (Ultimate)

The Ultimate Search Processor includes built-in workflow stage updates:

#### Workflow Stages
- **DOWNLOADING**: File download initiated
- **PROCESSING**: Document processing started
- **IMAGE_PROCESSING**: Image preprocessing in progress
- **OCR_EXTRACTION**: Text extraction in progress
- **TEXT_NORMALIZATION**: Text normalization in progress
- **VISION_ANALYSIS**: Visual analysis in progress
- **FINALIZATION**: Processing finalization
- **COMPLETED**: Processing completed successfully
- **FAILED**: Processing failed with error

## Docker Commands

### Standard Pipeline
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and start
docker-compose up --build -d

# Clean up (removes volumes)
docker-compose down -v
```

### Ultimate Search Processor
```bash
# Navigate to ultimate directory
cd ultimate

# Start Ultimate services
./start.sh

# Stop services
docker compose -f docker-compose.ultimate.yml down

# View logs
docker logs vector-storage-processing-service-python-ultimate-search-1
docker logs vector-storage-processing-service-python-ultimate-celery-worker-1
```

## Performance Optimization

### OCR Performance (Ultimate)
- **Image Preprocessing**: Multiple techniques applied automatically
- **Batch Processing**: Process multiple files simultaneously
- **Caching**: Results cached for repeated searches
- **Timeout Protection**: 30-second processing limit per file

### Memory Management
- **Streaming Processing**: Large files processed in chunks
- **Model Optimization**: CPU-optimized models for better performance
- **Resource Limits**: Automatic cleanup of temporary files

### Recommended Settings
```python
# For high-volume processing
MAX_CONCURRENT_FILES = 5
PROCESSING_TIMEOUT = 30
CACHE_SIZE = 1000
```

## Troubleshooting

### Container Health Issues

If containers fail health checks:

1. **Check logs**:
```bash
# Standard pipeline
docker-compose logs [service-name]

# Ultimate Search Processor
docker logs vector-storage-processing-service-python-ultimate-search-1
```

2. **Restart specific service**:
```bash
# Standard pipeline
docker-compose restart [service-name]

# Ultimate Search Processor
docker compose -f ultimate/docker-compose.ultimate.yml restart [service-name]
```

3. **Full restart**:
```bash
# Standard pipeline
docker-compose down && docker-compose up -d

# Ultimate Search Processor
cd ultimate && docker compose -f docker-compose.ultimate.yml down && docker compose -f docker-compose.ultimate.yml up -d
```

### Common Issues

- **Port conflicts**: Ensure ports 8000, 8080, 19530, 6380 are available
- **Memory issues**: Increase Docker memory allocation to at least 4GB
- **Model download failures**: Check internet connection and retry
- **Milvus connection**: Wait for Milvus to fully initialize (can take 2-3 minutes)
- **Tesseract not found**: Ensure Tesseract is installed and `TESSERACT_CMD` is set correctly

### Reset Everything

```bash
# Standard pipeline
docker-compose down -v
docker system prune -f
docker-compose up -d

# Ultimate Search Processor
cd ultimate
docker compose -f docker-compose.ultimate.yml down -v
docker system prune -f
./start.sh
```

## Monitoring

- **Container Status**: `docker-compose ps` or `docker ps`
- **Resource Usage**: `docker stats`
- **Application Logs**: `docker-compose logs -f api` or `docker logs <container_id>`
- **Milvus Logs**: `docker-compose logs -f milvus`

## Security Considerations

### File Upload Security
- File type validation
- Size limits (configurable)
- Temporary file cleanup
- No persistent storage of uploaded files

### API Security
- Input validation and sanitization
- Rate limiting (configurable)
- Error handling without information disclosure

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd vector-storage-processing-service-python

# Install development dependencies
pip install -r requirements.txt
pip install -r ultimate/requirements_ultimate.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black src/ ultimate/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review container logs
3. Open an issue on GitHub