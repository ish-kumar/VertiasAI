"""
FastAPI Server - Main Entry Point

This starts the FastAPI server with:
- Document upload/management endpoints
- Query submission endpoint
- Stats endpoint
- CORS middleware for frontend

Usage:
    uvicorn main:app --reload --port 8000
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from fastapi import FastAPI
from loguru import logger

# Import API factory
from src.api.main import create_app

# Import pipeline and routes
from src.ingestion.pipeline import IngestionPipeline
from src.agents.retriever import initialize_vector_store
from src.graph.state_machine import get_compiled_graph
from src.api.routes import documents, query, stats
from src.retrieval.pgvector_store import PGVectorStore
from src.utils.config import get_settings
from src.utils.supabase_client import validate_supabase_ready


# Initialize application
logger.info("Initializing Legal RAG API...")

# Create FastAPI app
app = create_app()

# Global instances
pipeline = None
graph = None
_runtime_init_lock = asyncio.Lock()
_runtime_initialized = False


async def ensure_runtime_initialized() -> None:
    """
    Initialize heavy runtime components lazily.

    This avoids blocking server startup on cloud hosts where port binding
    must happen quickly (e.g., Render).
    """
    global pipeline, graph, _runtime_initialized
    if _runtime_initialized:
        return

    async with _runtime_init_lock:
        if _runtime_initialized:
            return

        logger.info("Initializing runtime components (lazy)...")
        settings = get_settings()

        if settings.vector_store_type == "pgvector":
            logger.info("Using pgvector mode (Supabase)")
            validate_supabase_ready()
            pipeline = IngestionPipeline(
                lazy_embedder=True,
                create_vector_store=False,
            )
            pg_store = PGVectorStore.from_settings()
            initialize_vector_store(pg_store, pipeline.embedder)
            logger.success("pgvector retrieval initialized")
        else:
            index_dir = Path("./vector_store")
            if index_dir.exists():
                logger.info(f"Loading existing index from {index_dir}")
                pipeline = IngestionPipeline.load_index(index_dir)
                logger.success(f"Loaded index with {pipeline.vector_store.index.ntotal} chunks")
            else:
                logger.info("No existing index found, creating new pipeline")
                pipeline = IngestionPipeline(lazy_embedder=True)
                logger.info("Empty pipeline created (upload documents to populate)")

            initialize_vector_store(pipeline.vector_store, pipeline.embedder)
            logger.success("FAISS retrieval initialized")

        graph = get_compiled_graph()
        logger.success("LangGraph compiled")

        documents.set_pipeline(pipeline)
        query.set_graph(graph)
        stats.set_pipeline(pipeline)
        _runtime_initialized = True
        logger.success("✅ Runtime components initialized")


@app.on_event("startup")
async def startup_event():
    """
    Initialize the system on startup.
    
    Steps:
    1. Load or create vector store
    2. Initialize retrieval agent
    3. Compile LangGraph
    4. Set global instances for routes
    """
    logger.info("=" * 80)
    logger.info("LEGAL RAG API STARTUP")
    logger.info("=" * 80)
    logger.success("API booted. Runtime components will initialize on first request.")
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up on shutdown.
    
    Saves the vector store index before shutting down.
    """
    global pipeline
    
    logger.info("Shutting down Legal RAG API...")

    settings = get_settings()
    if settings.vector_store_type == "faiss" and pipeline and pipeline.vector_store.index.ntotal > 0:
        try:
            index_dir = Path("./vector_store")
            pipeline.save_index(index_dir)
            logger.success(f"Saved index to {index_dir}")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    logger.info("Shutdown complete")


# For development/testing
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
