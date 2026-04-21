"""
Document Ingestion Pipeline - End-to-end document processing.

This orchestrates the full flow:
1. Parse document (PDF/DOCX/TXT)
2. Chunk into semantic pieces
3. Generate embeddings
4. Add to vector store

Usage:
    pipeline = IngestionPipeline()
    pipeline.ingest_document("contract.pdf", document_id="DOC001")
    pipeline.save_index("./vector_store")
"""

from pathlib import Path
from typing import Optional, List
from loguru import logger

from .document_parser import DocumentParser
from .chunker import LegalChunker, DocumentChunk
from .embedder import EmbeddingGenerator
from ..retrieval.vector_store import FAISSVectorStore


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.
    
    Components:
    - DocumentParser: Extracts text from files
    - LegalChunker: Splits into semantic chunks
    - EmbeddingGenerator: Creates vector embeddings
    - FAISSVectorStore: Indexes for search
    
    Why pipeline pattern:
    - Orchestrates complex workflow
    - Each component is swappable
    - Easy to test each stage
    - Clear data flow
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        vector_store: Optional[FAISSVectorStore] = None,
        lazy_embedder: bool = False,
        create_vector_store: bool = True,
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            embedding_model: Sentence-transformers model name
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            vector_store: Optional existing vector store (or create new)
        """
        logger.info("Initializing document ingestion pipeline")
        
        # Initialize components
        self.parser = DocumentParser()
        self.chunker = LegalChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedder = EmbeddingGenerator(
            model_name=embedding_model,
            lazy_load=lazy_embedder,
        )

        # Initialize or use provided vector store.
        # In pgvector mode we can skip FAISS store creation entirely.
        if not create_vector_store:
            self.vector_store = vector_store
        elif vector_store is None:
            embedding_dim = self.embedder.get_embedding_dimension()
            self.vector_store = FAISSVectorStore(embedding_dim=embedding_dim)
        else:
            self.vector_store = vector_store
        
        logger.success("Ingestion pipeline initialized")
    
    def ingest_document(
        self,
        file_path: str | Path,
        document_id: str,
        metadata: Optional[dict] = None
    ) -> List[DocumentChunk]:
        """
        Ingest a single document through the full pipeline.
        
        Args:
            file_path: Path to document file
            document_id: Unique identifier for this document
            metadata: Additional metadata (jurisdiction, doc_type, etc.)
            
        Returns:
            List of created DocumentChunks
            
        Pipeline stages:
        1. Parse → ParsedDocument
        2. Chunk → List[DocumentChunk]
        3. Embed → np.ndarray
        4. Index → FAISSVectorStore
        """
        logger.info(f"Ingesting document: {document_id} from {file_path}")
        
        metadata = metadata or {}
        
        # Stage 1: Parse document
        parsed_doc = self.parser.parse_file(file_path)
        logger.info(f"Parsed {len(parsed_doc.text)} characters")
        
        # Merge metadata
        full_metadata = {**parsed_doc.metadata, **metadata}
        
        # Stage 2: Chunk document
        chunks = self.chunker.chunk_document(
            text=parsed_doc.text,
            document_id=document_id,
            metadata=full_metadata
        )
        logger.info(f"Created {len(chunks)} chunks")
        
        if not chunks:
            logger.warning(f"No chunks created for {document_id}")
            return []
        
        # Stage 3: Generate embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_texts(chunk_texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Stage 4: Add to vector store
        self.vector_store.add_chunks(chunks, embeddings)
        logger.success(
            f"Indexed {len(chunks)} chunks for document {document_id}"
        )
        
        return chunks
    
    def ingest_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        supported_extensions: Optional[List[str]] = None
    ) -> int:
        """
        Ingest all documents in a directory.
        
        Args:
            directory: Path to directory
            recursive: Whether to search subdirectories
            supported_extensions: File extensions to process
            
        Returns:
            Number of documents ingested
            
        Use case: Bulk ingestion of document library
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        supported_extensions = supported_extensions or [".pdf", ".docx", ".txt"]
        
        # Find all matching files
        if recursive:
            files = []
            for ext in supported_extensions:
                files.extend(directory.rglob(f"*{ext}"))
        else:
            files = []
            for ext in supported_extensions:
                files.extend(directory.glob(f"*{ext}"))
        
        logger.info(f"Found {len(files)} documents in {directory}")
        
        # Ingest each file
        ingested_count = 0
        for file_path in files:
            try:
                # Generate document ID from filename
                document_id = file_path.stem  # Filename without extension
                
                # Extract jurisdiction from filename if present
                # e.g., "employment_contract_CA.pdf" → {"jurisdiction": "CA"}
                metadata = self._extract_metadata_from_filename(file_path.name)
                
                self.ingest_document(file_path, document_id, metadata)
                ingested_count += 1
                
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
                continue
        
        logger.success(
            f"Ingested {ingested_count}/{len(files)} documents"
        )
        
        return ingested_count
    
    @staticmethod
    def _extract_metadata_from_filename(filename: str) -> dict:
        """
        Extract metadata from filename patterns.
        
        Patterns:
        - employment_agreement_CA.pdf → {"doc_type": "employment_agreement", "jurisdiction": "CA"}
        - contract_2024_NY.docx → {"doc_type": "contract", "year": "2024", "jurisdiction": "NY"}
        
        Why filename conventions:
        - Simple way to add metadata
        - No need for separate metadata files
        - User-friendly
        """
        metadata = {}
        
        # Common patterns
        # This is a simple example - customize for your naming conventions
        lower_filename = filename.lower()
        
        # Detect document type
        if "employment" in lower_filename:
            metadata["doc_type"] = "employment_agreement"
        elif "nda" in lower_filename or "confidential" in lower_filename:
            metadata["doc_type"] = "nda"
        elif "contract" in lower_filename:
            metadata["doc_type"] = "contract"
        
        # Detect jurisdiction (state codes)
        # e.g., _CA, _NY, _TX
        import re
        state_match = re.search(r'_([A-Z]{2})(?:\.|_|$)', filename)
        if state_match:
            metadata["jurisdiction"] = state_match.group(1)
        
        # Detect year
        year_match = re.search(r'_(\d{4})(?:\.|_|$)', filename)
        if year_match:
            metadata["year"] = int(year_match.group(1))
        
        return metadata
    
    def save_index(self, directory: str | Path):
        """
        Save the vector store index to disk.
        
        Args:
            directory: Directory to save to
        """
        self.vector_store.save(directory)
    
    @classmethod
    def load_index(
        cls,
        directory: str | Path,
        embedding_model: str = "all-MiniLM-L6-v2"
    ) -> "IngestionPipeline":
        """
        Load an existing index from disk.
        
        Args:
            directory: Directory to load from
            embedding_model: Model to use for new embeddings
            
        Returns:
            IngestionPipeline with loaded index
        """
        logger.info(f"Loading index from {directory}")
        
        # Load vector store
        vector_store = FAISSVectorStore.load(directory)
        
        # Create pipeline with loaded store
        pipeline = cls(
            embedding_model=embedding_model,
            vector_store=vector_store
        )
        
        logger.success(f"Loaded pipeline with {vector_store.index.ntotal} chunks")
        
        return pipeline
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            **self.vector_store.get_stats(),
            "embedding_model": self.embedder.model_name,
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap,
        }
