import yaml
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class RAGProvider:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.llm = None
        self.qa_chain = None
        self.db = None  # Store database reference for category filtering
        
    def _load_config(self, path):
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def initialize(self):
        print("Initializing RAG Provider...")
        # 1. Init Embeddings & Vector Store
        embedding_model = self.config.get("embedding_model_name", "all-MiniLM-L6-v2")
        persist_dir = self.config.get("persist_directory", "chroma_db")
        
        print(f"Loading embeddings: {embedding_model}")
        embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
        
        print(f"Loading ChromaDB from {persist_dir}")
        self.db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # Retrieve top 10 most relevant chunks
        )
        
        # 2. Init LLM
        model_path = self.config.get("model_path")
        if not model_path or not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}. QA will fail.")
            # We allow initialization to proceed so ingestion can work even without model
        else:
            print(f"Loading LLM from {model_path}...")
            # Adjust n_gpu_layers if you have a GPU
            if not self.llm:
                self.llm = LlamaCpp(
                    model_path=model_path,
                    n_ctx=self.config.get("n_ctx", 2048),
                    n_threads=self.config.get("n_threads", 4),
                    temperature=0.3,  # Increased for better response quality
                    max_tokens=self.config.get("max_tokens", 2048),
                    request_timeout=self.config.get("request_timeout", 300),
                    verbose=True
                )
            
            # 3. Init Chain with Custom Prompt
            print("Creating RetrievalQA chain...")
            
            # Custom prompt template for better responses
            template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Provide a detailed and helpful answer based on the context.

Context: {context}

Question: {question}

Answer:"""
            
            PROMPT = PromptTemplate(
                template=template, 
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
        print("RAG Provider initialized.")

    def query(self, query_text: str, category: str = None):
        """Query the RAG system, optionally filtering by category.
        
        Args:
            query_text: The question to ask
            category: Optional category to filter documents (e.g., "Leave", "Medical")
                     Special: "Noting" category bypasses RAG and uses LLM directly
        """
        if not self.llm:
            return {"result": "Error: LLM not initialized (model not found?)", "source_documents": []}
        
        # Special case: "Noting" category = direct LLM chat (no RAG)
        if category and category.lower() == "noting":
            print(f"\n{'='*60}")
            print(f"Direct Chat Mode (No RAG)")
            print(f"Query: {query_text}")
            print(f"{'='*60}")
            
            # Call LLM directly without retrieval
            try:
                response = self.llm.invoke(query_text)
                print(f"LLM Response: {response[:200]}...")
                print(f"{'='*60}\n")
                
                return {
                    "result": response,
                    "source_documents": []  # No documents used
                }
            except Exception as e:
                return {
                    "result": f"Error calling LLM: {str(e)}",
                    "source_documents": []
                }
        
        # RAG mode: retrieval + LLM
        if not self.qa_chain:
            return {"result": "Error: QA chain not initialized", "source_documents": []}
        
        # If category filter is specified, create a filtered retriever
        if category:
            print(f"Filtering search to category: {category}")
            filtered_retriever = self.db.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 10,
                    "filter": {"category": category}
                }
            )
            # Create a temporary QA chain with filtered retriever
            from langchain.prompts import PromptTemplate
            template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Provide a detailed and helpful answer based on the context.

Context: {context}

Question: {question}

Answer:"""
            PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
            
            from langchain.chains import RetrievalQA
            filtered_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=filtered_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            result = filtered_chain.invoke(query_text)
        else:
            result = self.qa_chain.invoke(query_text)
        
        # Post-processing to identify latest versions
        docs = result.get("source_documents", [])
        if docs:
            # Group by source/category to find latest versions in retrieved set
            latest_versions = {}
            for doc in docs:
                source = doc.metadata.get("source", "unknown")
                cat = doc.metadata.get("category", "General")
                version = doc.metadata.get("version", 1)
                key = (source, cat)
                if key not in latest_versions or version > latest_versions[key]:
                    latest_versions[key] = version
            
            # Tag docs as "Latest" or "Legacy"
            for doc in docs:
                source = doc.metadata.get("source", "unknown")
                cat = doc.metadata.get("category", "General")
                version = doc.metadata.get("version", 1)
                if version == latest_versions.get((source, cat)):
                    doc.metadata["is_latest"] = True
                else:
                    doc.metadata["is_latest"] = False
        
        # Debug logging to see what chunks were retrieved
        print(f"\n{'='*60}")
        print(f"Query: {query_text}")
        if category:
            print(f"Category Filter: {category}")
        print(f"{'='*60}")
        print(f"Retrieved {len(result.get('source_documents', []))} chunks:")
        for i, doc in enumerate(result.get('source_documents', []), 1):
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', '?')
            doc_category = doc.metadata.get('category', 'N/A')
            doc_version = doc.metadata.get('version', 1)
            is_latest = doc.metadata.get('is_latest', True)
            status = " [LATEST]" if is_latest else " [OLD VERSION]"
            print(f"  [{i}]{status} {source} - Page {page} (Category: {doc_category}, Version: {doc_version})")
            print(f"      Preview: {doc.page_content[:100].strip()}...")
        print(f"{'='*60}\n")
        
        return result
