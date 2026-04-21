"""
Query endpoint - Submit legal queries to the RAG pipeline.

Routes:
- POST /api/query - Submit a legal query and get response
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from loguru import logger

router = APIRouter()

# Global graph instance (set by main.py)
_graph = None


def set_graph(graph):
    """Set the global LangGraph instance."""
    global _graph
    _graph = graph


class QueryRequest(BaseModel):
    """Query request payload."""
    query: str
    jurisdiction: Optional[str] = None
    context: Optional[str] = None


@router.post("/")
async def submit_query(request: QueryRequest):
    """
    Submit a legal query to the RAG pipeline.
    
    This runs the full pipeline:
    1. Classify query
    2. Retrieve relevant clauses
    3. Generate answer (with LLM)
    4. Generate counter-arguments (with LLM)
    5. Verify citations
    6. Score confidence & risk
    7. Decision gate (answer vs refuse)
    8. Format response
    
    Returns:
        Complete RAG response with answer/refusal, confidence, risk, citations, etc.
    """
    if _graph is None:
        from main import ensure_runtime_initialized
        await ensure_runtime_initialized()
        if _graph is None:
            raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    
    try:
        logger.info(f"Received query: {request.query[:100]}...")
        
        # Import here to avoid circular imports
        from ...graph.state_machine import run_legal_rag_query
        
        # Run the full RAG pipeline
        response = await run_legal_rag_query(
            query_text=request.query,
            jurisdiction=request.jurisdiction,
            context=request.context
        )
        
        # Convert Pydantic models to dict for JSON response
        response_dict = response.dict() if hasattr(response, 'dict') else response
        
        logger.success(
            f"Query completed: {response_dict.get('query_id')} - "
            f"{'ANSWERED' if response_dict.get('success') else 'REFUSED'}"
        )
        
        return response_dict
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
