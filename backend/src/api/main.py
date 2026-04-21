"""
FastAPI Application - Main entry point.

This creates the FastAPI app with all routes and middleware.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .routes import documents, query, stats


def _get_allowed_origins() -> list[str]:
    """
    Read allowed origins from ALLOWED_ORIGINS env var (comma-separated).
    Always includes localhost for local development.
    """
    default = ["http://localhost:3000", "http://localhost:3001"]
    raw = os.environ.get("ALLOWED_ORIGINS", "")
    if not raw.strip():
        return default
    extra = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    return list(dict.fromkeys(extra + default))  # deduplicate, extra first


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Why this pattern:
    - Factory function (can create multiple instances)
    - Clean separation of concerns
    - Easy to test
    - Configurable
    """
    app = FastAPI(
        title="Legal RAG API",
        description="Production-grade Legal RAG system with adversarial reasoning",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )
    
    # CORS middleware
    origins = _get_allowed_origins()
    logger.info(f"CORS allowed origins: {origins}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
    app.include_router(query.router, prefix="/api/query", tags=["query"])
    app.include_router(stats.router, prefix="/api/stats", tags=["stats"])
    
    # Health check endpoint
    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "legal-rag-api"}
    
    logger.info("FastAPI application created")
    
    return app
