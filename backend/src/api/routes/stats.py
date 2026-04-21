"""
Stats endpoint - Get system statistics and health info.

Routes:
- GET /api/stats - Get system statistics
"""

from fastapi import APIRouter, HTTPException
from loguru import logger
from ...utils.config import get_settings
from ...utils.supabase_client import list_documents_with_chunk_counts

router = APIRouter()

# Global pipeline instance (set by main.py)
_pipeline = None


def set_pipeline(pipeline):
    """Set the global pipeline instance."""
    global _pipeline
    _pipeline = pipeline


@router.get("/")
async def get_stats():
    """
    Get system statistics.
    
    Returns:
        - Total documents indexed
        - Total chunks
        - Vector store stats
        - Embedding model info
        - Memory usage
    """
    if _pipeline is None:
        from main import ensure_runtime_initialized
        await ensure_runtime_initialized()
        if _pipeline is None:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        settings = get_settings()
        if settings.vector_store_type == "pgvector":
            docs = list_documents_with_chunk_counts()
            total_chunks = sum(d.get("chunk_count", 0) for d in docs)
            return {
                "total_documents": len(docs),
                "total_chunks": total_chunks,
                "embedding_model": _pipeline.embedder.model_name,
                "embedding_dimension": _pipeline.embedder.embedding_dim,
                "chunk_size": _pipeline.chunker.chunk_size,
                "chunk_overlap": _pipeline.chunker.chunk_overlap,
                "index_type": "pgvector",
                "memory_usage_mb": None,
                "status": "healthy"
            }

        # Get pipeline stats
        stats = _pipeline.get_stats()
        
        # Get unique document count
        chunks = _pipeline.vector_store.chunks
        doc_count = len(set(chunk.document_id for chunk in chunks))
        
        # Calculate memory usage (rough estimate)
        memory_mb = stats.get('memory_bytes', 0) / (1024 * 1024)
        
        return {
            "total_documents": doc_count,
            "total_chunks": stats.get('total_chunks', 0),
            "embedding_model": stats.get('embedding_model', 'unknown'),
            "embedding_dimension": stats.get('embedding_dim', 0),
            "chunk_size": stats.get('chunk_size', 0),
            "chunk_overlap": stats.get('chunk_overlap', 0),
            "index_type": stats.get('index_type', 'unknown'),
            "memory_usage_mb": round(memory_mb, 2),
            "status": "healthy"
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
