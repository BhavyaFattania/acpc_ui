import os
import gc
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx

# Core LlamaIndex imports
from llama_index.core import Settings, StorageContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate

# Cloud / integration imports
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.cohere_rerank import CohereRerank

load_dotenv()

# ============================================================================
# 1. GLOBAL CONFIGURATION
# ============================================================================
DATA_DIR        = "root/"
COLLECTION_NAME = "ld_chatbot_main"
QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64
RERANKER_MODEL  = "rerank-english-v3.0"
COHERE_API_KEY  = os.getenv("COHERE_API_KEY")
SARVAM_API_KEY  = os.getenv("SARVAM_API_KEY")  # Add this to your .env file

# Custom prompt controls
USER_QUERY_PREFIX = os.getenv("USER_QUERY_PREFIX", "").strip()
TEXT_QA_PROMPT_TEMPLATE = PromptTemplate(
    os.getenv(
        "TEXT_QA_PROMPT_TEMPLATE",
        """You are an ACPC admissions assistant. Answer the user question using only the provided context.

Rules:
1. Be accurate and concise.
2. If the answer is not in context, clearly say you do not have that information.
3. Mention important dates, eligibility, documents, or deadlines exactly as present in context.
4. Do not fabricate facts.
5. Provide source link at the end of your answer if relevant information is found in context.

Context:
{context_str}

Question:
{query_str}

Answer:"""
    )
)

# ============================================================================
# 2. GLOBAL SETTINGS
# ============================================================================
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    api_base="https://openrouter.ai/api/v1"
)

Settings.llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    api_base="https://openrouter.ai/api/v1",
    is_chat_model=True,
    reuse_client=False,
    additional_kwargs={
        "extra_headers": {
            "HTTP-Referer": "https://ldce.ac.in",
            "X-Title": "LDCE Chatbot",
        }
    }
)


# ============================================================================
# 3. PIPELINE CLASSES (unchanged from working code)
# ============================================================================
class DataIngestionPipeline:
    def __init__(self, collection_name=COLLECTION_NAME):
        self.collection_name = collection_name
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
        self.vector_store = QdrantVectorStore(
            collection_name=self.collection_name,
            client=self.client,
            enable_hybrid=True
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.node_parser = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    def get_meta(self, file_path):
        file_name_with_ext = os.path.basename(file_path)
        base_name = os.path.splitext(file_name_with_ext)[0]
        url = f"https://acpc.gujarat.gov.in/assets/uploads/media-uploader/{base_name}.pdf"
        return {
            "file_name": f"{base_name}.pdf",
            "url": url,
            "page_id": url
        }

    def ingest_document(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        reader = SimpleDirectoryReader(
            input_files=[file_path],
            file_metadata=self.get_meta
        )
        documents = reader.load_data()
        nodes = self.node_parser.get_nodes_from_documents(documents)
        VectorStoreIndex(nodes, storage_context=self.storage_context)
        return {"success": True, "chunks": len(nodes)}

    def ingest_directory(self, directory_path=DATA_DIR, recursive=True):
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        reader = SimpleDirectoryReader(
            input_dir=directory_path,
            recursive=recursive,
            file_metadata=self.get_meta
        )
        documents = reader.load_data()
        nodes = self.node_parser.get_nodes_from_documents(documents)
        VectorStoreIndex(nodes, storage_context=self.storage_context)
        return {
            "success": True,
            "total_documents": len(documents),
            "total_chunks": len(nodes)
        }

    def reset(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.vector_store = QdrantVectorStore(
            collection_name=self.collection_name,
            client=self.client,
            enable_hybrid=True
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        return {"success": True, "message": f"Collection '{self.collection_name}' reset."}


class QueryResponsePipeline:
    def __init__(
        self,
        collection_name=COLLECTION_NAME,
        similarity_top_k=15,
        rerank_top_n=5,
        use_reranker=True,
    ):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
        self.vector_store = QdrantVectorStore(
            collection_name=collection_name,
            client=self.client,
            enable_hybrid=True
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        reranker = None
        if use_reranker:
            reranker = CohereRerank(
                model=RERANKER_MODEL,
                top_n=rerank_top_n,
                api_key=COHERE_API_KEY
            )

        index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context
        )
        retriever = index.as_retriever(similarity_top_k=similarity_top_k)

        if reranker:
            self.query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                node_postprocessors=[reranker],
                text_qa_template=TEXT_QA_PROMPT_TEMPLATE
            )
        else:
            self.query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                text_qa_template=TEXT_QA_PROMPT_TEMPLATE
            )

    def query(self, question: str):
        final_question = (
            f"{USER_QUERY_PREFIX}\n\n{question}" if USER_QUERY_PREFIX else question
        )
        response = self.query_engine.query(final_question)
        sources = []
        if hasattr(response, "source_nodes"):
            for i, node in enumerate(response.source_nodes, 1):
                sources.append({
                    "rank": i,
                    "score": round(node.score, 4) if node.score is not None else None,
                    "text_preview": node.text[:300] + "..." if len(node.text) > 300 else node.text,
                    "file_name": node.metadata.get("file_name", "N/A"),
                    "url": node.metadata.get("url", "N/A"),
                })
        return {
            "question": question,
            "answer": response.response,
            "sources": sources,
            "num_sources": len(sources)
        }


# ============================================================================
# 4. APP LIFESPAN — initialise pipelines once at startup
# ============================================================================
ingestion_pipeline: Optional[DataIngestionPipeline] = None
query_pipeline: Optional[QueryResponsePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ingestion_pipeline, query_pipeline
    ingestion_pipeline = DataIngestionPipeline()
    query_pipeline = QueryResponsePipeline()
    print("✅ Pipelines ready")
    yield
    # cleanup
    gc.collect()
    print("🛑 Shutting down")


# ============================================================================
# 5. FASTAPI APP
# ============================================================================
app = FastAPI(
    title="LDCE Chatbot API",
    description="RAG chatbot for LDCE institutional documents",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 6. REQUEST / RESPONSE MODELS
# ============================================================================
class QueryRequest(BaseModel):
    question: str

class IngestFileRequest(BaseModel):
    file_path: str

class IngestDirectoryRequest(BaseModel):
    directory_path: str = DATA_DIR
    recursive: bool = True


# ============================================================================
# 7. ROUTES
# ============================================================================
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "LDCE Chatbot API is running"}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "ingestion_pipeline": ingestion_pipeline is not None,
        "query_pipeline": query_pipeline is not None,
    }


@app.post("/query", tags=["Query"])
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = query_pipeline.query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stt", tags=["Speech-to-Text"])
async def speech_to_text(audio: UploadFile = File(...)):
    """
    Convert speech audio to text using Sarvam AI STT API
    Accepts audio file and returns transcribed text
    """
    if not SARVAM_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="SARVAM_API_KEY not configured in environment"
        )
    
    try:
        # Read the audio file
        audio_content = await audio.read()
        
        # Prepare the request to Sarvam API
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {
                'file': (audio.filename, audio_content, audio.content_type)
            }
            headers = {
                'api-subscription-key': SARVAM_API_KEY
            }
            
            # Call Sarvam STT API
            response = await client.post(
                'https://api.sarvam.ai/speech-to-text',
                files=files,
                headers=headers
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Sarvam API error: {response.text}"
                )
            
            result = response.json()
            return {
                "success": True,
                "transcript": result.get("transcript", ""),
                "language": result.get("language_code", "unknown")
            }
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Speech-to-text request timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT error: {str(e)}")


@app.post("/ingest/file", tags=["Ingestion"])
def ingest_file(request: IngestFileRequest):
    try:
        result = ingestion_pipeline.ingest_document(request.file_path)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/directory", tags=["Ingestion"])
def ingest_directory(request: IngestDirectoryRequest):
    try:
        result = ingestion_pipeline.ingest_directory(
            directory_path=request.directory_path,
            recursive=request.recursive
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/reset", tags=["Ingestion"])
def reset_collection():
    try:
        result = ingestion_pipeline.reset()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))