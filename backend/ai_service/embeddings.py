"""
Vector Embeddings and Semantic Search
Implements RAG (Retrieval-Augmented Generation) using ChromaDB
"""

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import os
import uuid


class EmbeddingService:
    """Manages document embeddings and semantic search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        
        # Load sentence transformer model
        print(f"Loading embedding model: {model_name}...")
        self.encoder = SentenceTransformer(model_name, device=device)
        
        # Initialize ChromaDB
        chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")
        print(f"Initializing ChromaDB at {chroma_path}...")
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        print(f"✅ Embedding service initialized")
    
    def add_documents(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]],
        collection_name: str = "default"
    ):
        """
        Add documents to the vector database
        
        Args:
            texts: List of text chunks to embed
            metadata: List of metadata dicts (one per text)
            collection_name: Name of the collection (e.g., "student_123")
        """
        # Get or create collection
        try:
            collection = self.client.get_collection(collection_name)
        except:
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": f"Documents for {collection_name}"}
            )
        
        # Generate embeddings
        embeddings = self.encoder.encode(texts, convert_to_numpy=True)
        
        # Generate IDs
        ids = [str(uuid.uuid4()) for _ in texts]
        
        # Add to collection
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadata,
            ids=ids
        )
        
        print(f"✅ Added {len(texts)} documents to {collection_name}")
    
    def search(
        self,
        query: str,
        collection_name: str = "default",
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using semantic similarity
        
        Args:
            query: Search query
            collection_name: Collection to search in
            top_k: Number of results to return
            
        Returns:
            List of result dicts with 'content', 'metadata', and 'score'
        """
        try:
            collection = self.client.get_collection(collection_name)
        except:
            print(f"⚠️  Collection {collection_name} not found")
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)[0]
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'score': 1 - results['distances'][0][i] if results['distances'] else 0  # Convert distance to similarity
                })
        
        return formatted_results
    
    def delete_collection(self, collection_name: str):
        """Delete an entire collection"""
        try:
            self.client.delete_collection(collection_name)
            print(f"✅ Deleted collection: {collection_name}")
        except:
            print(f"⚠️  Collection {collection_name} not found")
    
    def list_collections(self) -> List[str]:
        """List all available collections"""
        collections = self.client.list_collections()
        return [col.name for col in collections]
    
    def get_collection_count(self, collection_name: str) -> int:
        """Get number of documents in a collection"""
        try:
            collection = self.client.get_collection(collection_name)
            return collection.count()
        except:
            return 0


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for better retrieval
    
    Args:
        text: Text to chunk
        chunk_size: Target size of each chunk (in characters)
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < text_length:
            # Look for sentence ending
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks
