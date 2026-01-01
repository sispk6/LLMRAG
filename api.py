from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import shutil
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import glob
import logging
from rag_engine import RAGProvider
from ingest import ingest_documents
from fastapi.security import APIKeyHeader
from fastapi import Security


#COment
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EsDeeGee Offline Document Search AI Based", 
    version="1.0", 
    description="AI Search eSDeeGee with Local LLM developed by deekit",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    openapi_url="/Prism.AI",
    docs_url=None
)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    from fastapi.openapi.docs import get_swagger_ui_html
    from fastapi.responses import HTMLResponse
    
    html = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - API Docs",
        swagger_ui_parameters=app.swagger_ui_parameters,
    )
    
    # Replace OAS 3.1 text with Prism.AI using CSS
    custom_css = """
    <style>
        .swagger-ui .info .title .version-stamp .version { display: none; }
        .swagger-ui .info .title .version-stamp::after { 
            content: "Prism.AI"; 
            font-weight: bold;
        }
    </style>
    """
    
    content = html.body.decode().replace("</head>", f"{custom_css}</head>")
    return HTMLResponse(content=content)
@app.get("/")
def read_root():
    return {"message": "Welcome to the Offline SDG API. Go to /docs to test usage."}

# Global RAG Instance
rag = RAGProvider()

# API Key Configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    # Get the key from environment
    required_key = os.getenv("API_KEY")
    
    # If no key is set in environment, allow all requests
    if not required_key:
        return None
    
    # If key is set, validate it
    if api_key_header == required_key:
        return api_key_header
    else:
        raise HTTPException(
            status_code=403, 
            detail="Could not validate API Key"
        )

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up... Initializing RAG Engine.")
    if os.getenv("API_KEY"):
        logger.info("API Key protection is ENABLED.")
    else:
        logger.info("API Key protection is DISABLED (no API_KEY env var found).")
    rag.initialize()

class QueryRequest(BaseModel):
    query: str
    category: str = None  # Optional category filter

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.get("/health")
def health_check():
    # Basic check to see if RAG is ready
    status = "ok"
    if not hasattr(rag, 'db') or rag.db is None:
        status = "initializing or error"
    
    return {
        "status": status, 
        "service": "Offline RAG",
        "engine_ready": rag.db is not None if hasattr(rag, 'db') else False
    }

@app.get("/ping")
def ping():
    """Simple connectivity check."""
    return {"message": "pong"}

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest, api_key: str = Security(get_api_key)):
    logger.info(f"Received query: {request.query} (category: {request.category})")
    try:
        response = rag.query(request.query, category=request.category)
        sources = []
        for doc in response.get("source_documents", []):
            status = " [LATEST]" if doc.metadata.get('is_latest', True) else " [OLD VERSION]"
            source_str = f"{doc.metadata.get('source', 'unknown')} (Page {doc.metadata.get('page', 0)}, Category: {doc.metadata.get('category', 'N/A')}, Version: {doc.metadata.get('version', 1)}){status}"
            sources.append(source_str)
        return {"answer": response.get("result", ""), "sources": sources}
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
def list_categories(api_key: str = Security(get_api_key)):
    """List all available document categories."""
    try:
        source_dir = rag.config.get("source_documents_dir", "source_documents")
        if not os.path.exists(source_dir):
            return {"categories": []}
        
        categories = []
        # Scan subdirectories
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)
            if os.path.isdir(item_path):
                categories.append(item)
        
        # Add "General" if there are files in root
        root_files = [f for f in os.listdir(source_dir) 
                      if os.path.isfile(os.path.join(source_dir, f)) 
                      and f.lower().endswith(('.pdf', '.docx', '.txt'))]
        if root_files and "General" not in categories:
            categories.insert(0, "General")
        
        return {"categories": sorted(categories)}
    except Exception as e:
        logger.error(f"Failed to list categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
def list_documents(api_key: str = Security(get_api_key)):
    """List all documents with their categories."""
    try:
        source_dir = rag.config.get("source_documents_dir", "source_documents")
        if not os.path.exists(source_dir):
            return {"documents": []}
        
        documents = []
        
        # Root level files (General category)
        for ext in ['*.pdf', '*.docx', '*.txt']:
            for path in glob.glob(os.path.join(source_dir, ext)):
                filename = os.path.basename(path)
                # Try to extract version from filename for the listing
                from ingest import extract_version
                version = extract_version(filename)
                
                documents.append({
                    "filename": filename,
                    "category": "General",
                    "version": version,
                    "path": path,
                    "size_bytes": os.path.getsize(path)
                })
        
        # Categorized files
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)
            if not os.path.isdir(item_path):
                continue
            
            category = item
            for ext in ['*.pdf', '*.docx', '*.txt']:
                for path in glob.glob(os.path.join(item_path, ext)):
                    filename = os.path.basename(path)
                    from ingest import extract_version
                    version = extract_version(filename)
                    
                    documents.append({
                        "filename": filename,
                        "category": category,
                        "version": version,
                        "path": path,
                        "size_bytes": os.path.getsize(path)
                    })
        
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
def trigger_ingest(api_key: str = Security(get_api_key)):
    """Triggers the document ingestion process."""
    try:
        logger.info("Starting ingestion...")
        ingest_documents()
        logger.info("Ingestion done. Reloading Retriever...")
        rag.initialize() # Reloads retriever to see new docs (LLM remains cached)
        return {"status": "Ingestion complete and RAG updated"}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/clear")
def clear_database(api_key: str = Security(get_api_key)):
    """Clears the vector database completely."""
    try:
        persist_dir = rag.config.get("persist_directory", "chroma_db")
        
        if not os.path.exists(persist_dir):
            return {"status": "No database found to clear"}
        
        # Close the database connection first
        if hasattr(rag, 'db') and rag.db is not None:
            try:
                # ChromaDB doesn't have an explicit close, but we can clear the reference
                rag.db = None
                logger.info("Closed database connection")
            except Exception as e:
                logger.warning(f"Error closing database: {e}")
        
        # Now try to remove the directory
        try:
            shutil.rmtree(persist_dir)
            logger.info(f"Cleared database at {persist_dir}")
            return {
                "status": "Database cleared successfully",
                "message": "Please restart the container or call /ingest to reinitialize"
            }
        except OSError as e:
            if "Device or resource busy" in str(e) or "Errno 16" in str(e):
                return {
                    "status": "error",
                    "message": "Database is locked. Stop the container, delete chroma_db folder manually, then restart."
                }
            raise
    except Exception as e:
        logger.error(f"Failed to clear database: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: str = "General",
    version: int = None,
    api_key: str = Security(get_api_key)
):
    """Uploads a file to the source_documents directory, organized by category and optional version."""
    try:
        # Ensure source directory exists
        source_dir = rag.config.get("source_documents_dir", "source_documents")
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)
        
        # Create category subdirectory
        category_dir = os.path.join(source_dir, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
            logger.info(f"Created category directory: {category_dir}")
            
        # If version is explicitly provided, rename file to follow convention
        final_filename = file.filename
        if version is not None:
            name, ext = os.path.splitext(file.filename)
            # Remove existing _vN if any
            import re
            name = re.sub(r'_v\d+$', '', name)
            final_filename = f"{name}_v{version}{ext}"
            
        file_location = os.path.join(category_dir, final_filename)
        with open(file_location, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"File saved: {file_location} (category: {category}, version: {version or 'auto'})")
        return {
            "filename": final_filename, 
            "category": category,
            "version": version or "auto-extracted",
            "message": f"File uploaded successfully to category '{category}'. Call /ingest to process it."
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
