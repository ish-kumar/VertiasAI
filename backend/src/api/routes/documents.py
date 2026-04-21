"""
Document management endpoints.

Routes:
- POST /api/documents/upload - Upload and ingest a document
- GET /api/documents - List all documents  
- DELETE /api/documents/{doc_id} - Delete a document
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import shutil
from pathlib import Path
import numpy as np

from loguru import logger
from ...utils.config import get_settings
from ...utils.supabase_client import (
    validate_supabase_ready,
    upload_document_file,
    delete_document_file,
    upsert_document_record,
    list_documents_with_chunk_counts,
    get_document_record,
    delete_document_records,
    clear_chunks_for_document,
    insert_chunk_rows,
)

router = APIRouter()

# Global pipeline instance (set by main.py)
_pipeline = None


def set_pipeline(pipeline):
    """Set the global ingestion pipeline instance."""
    global _pipeline
    _pipeline = pipeline


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_id: str = None,
    jurisdiction: str = None,
):
    """
    Upload and ingest a document.
    
    This endpoint:
    1. Saves the uploaded file
    2. Parses it (PDF/DOCX/TXT)
    3. Chunks it (clause-aware)
    4. Embeds it (sentence-transformers)
    5. Indexes it (FAISS)
    
    Returns:
        Document metadata and chunk count
    """
    if _pipeline is None:
        from main import ensure_runtime_initialized
        await ensure_runtime_initialized()
        if _pipeline is None:
            raise HTTPException(status_code=500, detail="Ingestion pipeline not initialized")
    
    try:
        settings = get_settings()
        # Generate document ID if not provided
        if not document_id:
            document_id = Path(file.filename).stem
        
        # Save uploaded file temporarily
        temp_dir = Path("./temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Uploaded file saved: {file_path}")
        
        # Prepare metadata
        metadata = {}
        if jurisdiction:
            metadata["jurisdiction"] = jurisdiction

        if settings.vector_store_type == "pgvector":
            validate_supabase_ready()

            # Read uploaded file bytes for storage upload
            file_bytes = file_path.read_bytes()
            storage_path = upload_document_file(
                file_bytes=file_bytes,
                original_filename=file.filename,
                document_id=document_id,
            )

            # Parse/chunk/embed using existing pipeline components
            parsed_doc = _pipeline.parser.parse_file(file_path)
            full_metadata = {**parsed_doc.metadata, **metadata}
            chunks = _pipeline.chunker.chunk_document(
                text=parsed_doc.text,
                document_id=document_id,
                metadata=full_metadata,
            )

            if chunks:
                chunk_texts = [c.text for c in chunks]
                embeddings = _pipeline.embedder.embed_texts(chunk_texts)
            else:
                embeddings = np.array([])

            # Upsert document row and refresh chunk rows for this document
            upsert_document_record(
                document_id=document_id,
                filename=file.filename,
                jurisdiction=jurisdiction,
                storage_path=storage_path,
            )
            clear_chunks_for_document(document_id)

            rows = []
            for chunk, emb in zip(chunks, embeddings):
                rows.append(
                    {
                        "document_id": chunk.document_id,
                        "clause_id": chunk.chunk_id,
                        "section": chunk.section,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "embedding": [float(x) for x in emb.tolist()],
                    }
                )
            insert_chunk_rows(rows)
        else:
            # FAISS path (existing behavior)
            chunks = _pipeline.ingest_document(
                file_path=file_path,
                document_id=document_id,
                metadata=metadata
            )
        
        # Clean up temp file
        file_path.unlink()
        
        return {
            "success": True,
            "document_id": document_id,
            "filename": file.filename,
            "chunks_created": len(chunks),
            "message": f"Successfully ingested {file.filename}"
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_documents():
    """
    List all indexed documents.
    
    Returns:
        List of document metadata
    """
    if _pipeline is None:
        from main import ensure_runtime_initialized
        await ensure_runtime_initialized()
        if _pipeline is None:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        settings = get_settings()
        if settings.vector_store_type == "pgvector":
            documents = list_documents_with_chunk_counts()
            return {
                "total_documents": len(documents),
                "documents": documents
            }

        # Get unique document IDs from chunks
        chunks = _pipeline.vector_store.chunks
        
        doc_ids = set(chunk.document_id for chunk in chunks)
        
        documents = []
        for doc_id in doc_ids:
            doc_chunks = [c for c in chunks if c.document_id == doc_id]
            
            # Get metadata from first chunk
            metadata = doc_chunks[0].metadata if doc_chunks else {}
            
            documents.append({
                "document_id": doc_id,
                "chunk_count": len(doc_chunks),
                "jurisdiction": metadata.get("jurisdiction", "Unknown"),
                "doc_type": metadata.get("doc_type", "Unknown"),
            })
        
        return {
            "total_documents": len(documents),
            "documents": documents
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document from the index.
    
    This properly removes the document by:
    1. Filtering out chunks from the document
    2. Re-embedding remaining chunks  
    3. Rebuilding the FAISS index
    
    Note: This operation can take a few seconds for large indices.
    """
    if _pipeline is None:
        from main import ensure_runtime_initialized
        await ensure_runtime_initialized()
        if _pipeline is None:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        settings = get_settings()

        if settings.vector_store_type == "pgvector":
            validate_supabase_ready()

            doc = get_document_record(document_id)
            if not doc:
                raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

            chunks_deleted = delete_document_records(document_id)
            storage_path = doc.get("storage_path")
            if storage_path:
                try:
                    delete_document_file(storage_path)
                except Exception as storage_err:
                    # Non-fatal for DB deletion success
                    logger.warning(f"Failed to delete storage file {storage_path}: {storage_err}")

            return {
                "success": True,
                "document_id": document_id,
                "chunks_deleted": chunks_deleted,
                "remaining_chunks": None,
                "message": f"Deleted {document_id} from Supabase storage and pgvector tables"
            }

        # Count chunks before
        chunks_before = len(_pipeline.vector_store.chunks)
        
        # Filter out chunks from this document
        remaining_chunks = [
            c for c in _pipeline.vector_store.chunks 
            if c.document_id != document_id
        ]
        
        chunks_deleted = chunks_before - len(remaining_chunks)
        
        if chunks_deleted == 0:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        logger.info(f"Deleting {chunks_deleted} chunks for {document_id}")
        
        # Rebuild the FAISS index without the deleted chunks
        if remaining_chunks:
            # Extract text from remaining chunks
            texts = [chunk.text for chunk in remaining_chunks]
            
            # Re-embed all remaining chunks
            logger.info(f"Re-embedding {len(remaining_chunks)} chunks...")
            embeddings = _pipeline.embedder.embed_texts(texts)
            
            # Create new FAISS index
            logger.info("Rebuilding FAISS index...")
            from ...retrieval.vector_store import FAISSVectorStore
            new_vector_store = FAISSVectorStore(
                embedding_dim=_pipeline.embedder.embedding_dim
            )
            new_vector_store.add_chunks(remaining_chunks, embeddings)
            
            # Replace the old vector store
            _pipeline.vector_store = new_vector_store
            
            # Update the retriever's global reference
            from ...agents.retriever import initialize_vector_store
            initialize_vector_store(_pipeline.vector_store, _pipeline.embedder)
            
            logger.success(f"Successfully deleted {document_id} and rebuilt index")
        else:
            # All chunks deleted - reset to empty index
            logger.info("All chunks deleted, creating empty index")
            from ...retrieval.vector_store import FAISSVectorStore
            _pipeline.vector_store = FAISSVectorStore(
                embedding_dim=_pipeline.embedder.embedding_dim
            )
            from ...agents.retriever import initialize_vector_store
            initialize_vector_store(_pipeline.vector_store, _pipeline.embedder)
        
        return {
            "success": True,
            "document_id": document_id,
            "chunks_deleted": chunks_deleted,
            "remaining_chunks": len(remaining_chunks),
            "message": f"Deleted {document_id} and rebuilt index ({chunks_deleted} chunks removed)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))
