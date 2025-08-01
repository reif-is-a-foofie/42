"""Vector store for 42 using Qdrant."""

from typing import List, Optional, Dict, Any
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from .interfaces import SearchResult, Chunk
from .config import load_config


class VectorStore:
    """Wraps Qdrant for vector storage and search."""
    
    def __init__(self, collection_name: str = None):
        """Initialize the vector store."""
        config = load_config()
        self.collection_name = collection_name or config.collection_name
        self.client = None
        self._connect()
    
    def _connect(self) -> None:
        """Connect to Qdrant."""
        try:
            config = load_config()
            self.client = QdrantClient(
                host=config.qdrant_host,
                port=config.qdrant_port
            )
            logger.info(f"Connected to Qdrant at {config.qdrant_host}:{config.qdrant_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def create_collection(self, vector_size: int) -> None:
        """Create the collection if it doesn't exist."""
        try:
            collections = self.client.get_collections()
            if self.collection_name not in [c.name for c in collections.collections]:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def upsert(self, points: List[PointStruct]) -> None:
        """Upsert points into the collection."""
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} points")
        except Exception as e:
            logger.error(f"Failed to upsert points: {e}")
            raise
    
    def search(self, query_vector: List[float], limit: int = 10) -> List[SearchResult]:
        """Search for similar vectors."""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            search_results = []
            for result in results:
                search_results.append(SearchResult(
                    text=result.payload.get("text", ""),
                    file_path=result.payload.get("file_path", ""),
                    score=result.score,
                    metadata=result.payload
                ))
            
            return search_results
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            raise
    
    def get_all_vectors(self) -> List[Dict[str, Any]]:
        """Get all vectors from the collection."""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust as needed
                with_vectors=True  # Include the actual vectors
            )
            
            vectors_data = []
            for point in results[0]:
                data = point.payload.copy()
                data['vector'] = point.vector  # Add the actual vector
                data['id'] = point.id  # Add the point ID
                vectors_data.append(data)
            
            return vectors_data
        except Exception as e:
            logger.error(f"Failed to get all vectors: {e}")
            raise
    
    def update_payload(self, point_id: str, payload: Dict[str, Any]) -> None:
        """Update payload for a specific point."""
        try:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=payload,
                points=[point_id]
            )
            logger.info(f"Updated payload for point {point_id}")
        except Exception as e:
            logger.error(f"Failed to update payload: {e}")
            raise
    
    def delete_collection(self) -> None:
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test connection to Qdrant."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
    
    def get_total_points(self) -> int:
        """Get total number of points in collection."""
        try:
            # Use direct HTTP call to avoid validation issues
            import httpx
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"http://{self.client._client._host}:{self.client._client._port}/collections/{self.collection_name}")
                if response.status_code == 200:
                    data = response.json()
                    return data.get("result", {}).get("points_count", 0)
                else:
                    logger.error(f"Failed to get collection info: {response.status_code}")
                    return 0
        except Exception as e:
            logger.error(f"Failed to get total points: {e}")
            return 0
    
    def count(self, collection_name: str = None) -> int:
        """Get count of points in collection (alias for get_total_points)."""
        return self.get_total_points()
    
    def search_by_url(self, url: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for points by URL in payload."""
        try:
            # Create a filter to search by URL
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="url",
                        match=MatchValue(value=url)
                    )
                ]
            )
            
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=limit
            )
            
            return [point.payload for point in results[0]]
        except Exception as e:
            logger.error(f"Failed to search by URL: {e}")
            return []
    
    def search_semantic(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for semantically similar documents using vector similarity."""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True
            )
            
            # Convert to list of dictionaries with payload
            similar_docs = []
            for result in results:
                doc = result.payload.copy()
                doc['similarity_score'] = result.score
                similar_docs.append(doc)
            
            logger.info(f"Found {len(similar_docs)} semantically similar documents")
            return similar_docs
            
        except Exception as e:
            logger.error(f"Failed to search semantically: {e}")
            return [] 