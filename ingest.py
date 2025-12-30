import os
import yaml
import glob
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# Load config
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("config.yaml not found. Using defaults.")
    config = {}

SOURCE_DIRECTORY = config.get("source_documents_dir", "source_documents")
PERSIST_DIRECTORY = config.get("persist_directory", "chroma_db")
EMBEDDING_MODEL_NAME = config.get("embedding_model_name", "all-MiniLM-L6-v2")
CHUNK_SIZE = config.get("chunk_size", 1000)
CHUNK_OVERLAP = config.get("chunk_overlap", 200)

def load_documents(source_dir: str) -> List[Document]:
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        
    documents = []
    
    # First, scan for files directly in source_dir (uncategorized/legacy files)
    for ext, loader_class in [('*.pdf', PyPDFLoader), ('*.docx', Docx2txtLoader), ('*.txt', TextLoader)]:
        for path in glob.glob(os.path.join(source_dir, ext)):
            try:
                if loader_class == TextLoader:
                    loader = loader_class(path, encoding="utf-8")
                else:
                    loader = loader_class(path)
                docs = loader.load()
                # Add category metadata (default to "General" for root-level files)
                for doc in docs:
                    doc.metadata["category"] = "General"
                documents.extend(docs)
                print(f"Loaded: {path} (category: General)")
            except Exception as e:
                print(f"Error loading {path}: {e}")
    
    # Second, scan subdirectories for categorized documents
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        if not os.path.isdir(item_path):
            continue
        
        category = item  # Directory name is the category
        
        # Load PDFs in this category
        for path in glob.glob(os.path.join(item_path, "*.pdf")):
            try:
                loader = PyPDFLoader(path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["category"] = category
                documents.extend(docs)
                print(f"Loaded: {path} (category: {category})")
            except Exception as e:
                print(f"Error loading {path}: {e}")

        # Load DOCX in this category
        for path in glob.glob(os.path.join(item_path, "*.docx")):
            try:
                loader = Docx2txtLoader(path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["category"] = category
                documents.extend(docs)
                print(f"Loaded: {path} (category: {category})")
            except Exception as e:
                print(f"Error loading {path}: {e}")
                
        # Load TXT in this category
        for path in glob.glob(os.path.join(item_path, "*.txt")):
            try:
                loader = TextLoader(path, encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["category"] = category
                documents.extend(docs)
                print(f"Loaded: {path} (category: {category})")
            except Exception as e:
                print(f"Error loading {path}: {e}")

    return documents

def ingest_documents():
    print(f"Loading documents from {SOURCE_DIRECTORY}...")
    documents = load_documents(SOURCE_DIRECTORY)
    
    if not documents:
        print("No documents found to ingest.")
        return

    print(f"Splitting {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks.")

    print(f"Creating embeddings using {EMBEDDING_MODEL_NAME}...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print(f"Storing in ChromaDB at {PERSIST_DIRECTORY}...")
    db = Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory=PERSIST_DIRECTORY
    )
    db.persist()
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_documents()
