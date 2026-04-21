"""
Embedding Generator - Converts text chunks to vector embeddings.

Strategy: Use sentence-transformers for fast, local embeddings.

Why sentence-transformers:
✅ Free (no API costs)
✅ Fast (runs locally, no network latency)
✅ Good quality (SOTA open-source models)
✅ Privacy-preserving (no data sent to external APIs)

Alternative: OpenAI embeddings
- Higher quality
- Costs money ($0.02 per 1M tokens)
- Network latency
- Good for production with budget
"""

from typing import List
import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed")


class EmbeddingGenerator:
    """
    Generates vector embeddings for text chunks.
    
    Model options:
    - all-MiniLM-L6-v2: Fast, 384 dimensions, good for most use cases
    - all-mpnet-base-v2: Slower, 768 dimensions, higher quality
    - all-MiniLM-L12-v2: Middle ground
    
    Why all-MiniLM-L6-v2 as default:
    - Fast inference (~10ms per chunk on CPU)
    - Small model size (~90MB)
    - Good retrieval quality
    - Works well for legal text
    
    For production tuning:
    - Fine-tune on legal documents for better domain adaptation
    - Use larger model if quality is more important than speed
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", lazy_load: bool = False):
        """
        Initialize embedding generator.
        
        Args:
            model_name: HuggingFace model name
            
        Why singleton pattern:
        - Model loading is expensive (~1-2 seconds)
        - Keep model in memory for entire session
        - Share across all embedding requests
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None

        if lazy_load:
            logger.info(
                f"Embedding model '{model_name}' set to lazy-load mode "
                "(will load on first embed request)"
            )
        else:
            self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        """Load sentence-transformers model on demand."""
        if self.model is not None:
            return

        logger.info(f"Loading embedding model: {self.model_name}")
        # Load model (downloads on first use, cached after)
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.success(
            f"Loaded {self.model_name}, embedding dimension: {self.embedding_dim}"
        )
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding (larger = faster but more memory)
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
            
        Why batching:
        - More efficient than one-by-one
        - Utilizes GPU/CPU better
        - Typical: 32-64 for CPU, 128-256 for GPU
        
        Performance:
        - CPU: ~100-200 texts/second (all-MiniLM-L6-v2)
        - GPU: ~1000+ texts/second
        """
        if not texts:
            return np.array([])

        self._ensure_model_loaded()
        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")
        
        # Generate embeddings
        # show_progress_bar=True for visibility during indexing
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
        )
        
        logger.success(
            f"Generated embeddings: {embeddings.shape} "
            f"({embeddings.nbytes / 1024 / 1024:.2f} MB)"
        )
        
        return embeddings
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            numpy array of shape (embedding_dim,)
            
        Use case: Embedding query at search time
        """
        self._ensure_model_loaded()
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding
    
    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of the embeddings."""
        self._ensure_model_loaded()
        return self.embedding_dim
