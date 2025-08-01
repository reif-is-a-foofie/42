"""Tests for the clustering engine module."""

import pytest
import numpy as np
import sys
import os
from typing import List, Dict, Any

# Add the 42 directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "42"))

from cluster import ClusteringEngine


class TestClusteringEngine:
    """Test clustering engine functionality."""
    
    @pytest.fixture
    def clustering_engine(self):
        """Create a clustering engine instance for testing."""
        return ClusteringEngine()
    
    @pytest.fixture
    def sample_vectors(self):
        """Create sample vectors for testing."""
        # Create 3 clusters of vectors
        np.random.seed(42)
        
        # Cluster 1: centered around [1, 1, 1]
        cluster1 = np.random.normal([1, 1, 1], 0.1, (10, 3))
        
        # Cluster 2: centered around [5, 5, 5]
        cluster2 = np.random.normal([5, 5, 5], 0.1, (10, 3))
        
        # Cluster 3: centered around [10, 10, 10]
        cluster3 = np.random.normal([10, 10, 10], 0.1, (10, 3))
        
        # Combine all vectors
        all_vectors = np.vstack([cluster1, cluster2, cluster3])
        return all_vectors.tolist()
    
    def test_clustering_engine_initialization(self, clustering_engine):
        """Test that clustering engine can be initialized."""
        assert clustering_engine is not None
        assert hasattr(clustering_engine, 'hdbscan')
        assert hasattr(clustering_engine, 'umap')
    
    def test_recluster_vectors(self, clustering_engine, sample_vectors):
        """Test reclustering vectors with HDBSCAN."""
        # Mock vector store data
        vector_data = []
        for i, vector in enumerate(sample_vectors):
            vector_data.append({
                "id": i,
                "vector": vector,
                "payload": {
                    "text": f"Sample text {i}",
                    "file_path": f"file_{i}.txt"
                }
            })
        
        # Test reclustering
        result = clustering_engine.recluster_vectors(vector_data)
        
        assert isinstance(result, dict)
        assert "clusters" in result
        assert "cluster_count" in result
        assert "clustered_points" in result
        
        # Should find 3 clusters
        assert result["cluster_count"] == 3
        assert len(result["clusters"]) == 3
        assert len(result["clustered_points"]) == len(sample_vectors)
    
    def test_cluster_vectors_with_metadata(self, clustering_engine, sample_vectors):
        """Test clustering with metadata preservation."""
        # Create vector data with metadata
        vector_data = []
        for i, vector in enumerate(sample_vectors):
            vector_data.append({
                "id": i,
                "vector": vector,
                "payload": {
                    "text": f"Sample text {i}",
                    "file_path": f"file_{i}.txt",
                    "metadata": {
                        "type": "text" if i < 10 else "code",
                        "language": "python"
                    }
                }
            })
        
        # Test clustering
        result = clustering_engine.recluster_vectors(vector_data)
        
        assert isinstance(result, dict)
        assert "clusters" in result
        
        # Check that metadata is preserved
        for point in result["clustered_points"]:
            assert "payload" in point
            assert "metadata" in point["payload"]
            assert "type" in point["payload"]["metadata"]
    
    def test_cluster_quality_metrics(self, clustering_engine, sample_vectors):
        """Test cluster quality metrics."""
        vector_data = []
        for i, vector in enumerate(sample_vectors):
            vector_data.append({
                "id": i,
                "vector": vector,
                "payload": {"text": f"Sample text {i}"}
            })
        
        result = clustering_engine.recluster_vectors(vector_data)
        
        # Check quality metrics
        assert "silhouette_score" in result or "cluster_quality" in result
        assert result["cluster_count"] > 0
        assert result["cluster_count"] <= len(sample_vectors)
    
    def test_empty_vectors_handling(self, clustering_engine):
        """Test handling of empty vector list."""
        result = clustering_engine.recluster_vectors([])
        
        assert isinstance(result, dict)
        assert result["cluster_count"] == 0
        assert len(result["clusters"]) == 0
    
    def test_single_vector_handling(self, clustering_engine):
        """Test handling of single vector."""
        vector_data = [{
            "id": 0,
            "vector": [1.0, 2.0, 3.0],
            "payload": {"text": "Single vector"}
        }]
        
        result = clustering_engine.recluster_vectors(vector_data)
        
        assert isinstance(result, dict)
        assert result["cluster_count"] == 1
        assert len(result["clusters"]) == 1
    
    def test_cluster_visualization(self, clustering_engine, sample_vectors):
        """Test cluster visualization generation."""
        vector_data = []
        for i, vector in enumerate(sample_vectors):
            vector_data.append({
                "id": i,
                "vector": vector,
                "payload": {"text": f"Sample text {i}"}
            })
        
        # Test with visualization
        result = clustering_engine.recluster_vectors(
            vector_data, 
            generate_plot=True,
            plot_path="test_clusters.png"
        )
        
        assert isinstance(result, dict)
        assert "plot_path" in result or "visualization" in result
    
    def test_cluster_parameters(self, clustering_engine, sample_vectors):
        """Test clustering with different parameters."""
        vector_data = []
        for i, vector in enumerate(sample_vectors):
            vector_data.append({
                "id": i,
                "vector": vector,
                "payload": {"text": f"Sample text {i}"}
            })
        
        # Test with custom parameters
        result = clustering_engine.recluster_vectors(
            vector_data,
            min_cluster_size=3,
            min_samples=2
        )
        
        assert isinstance(result, dict)
        assert "cluster_count" in result
    
    def test_cluster_labels(self, clustering_engine, sample_vectors):
        """Test that cluster labels are assigned correctly."""
        vector_data = []
        for i, vector in enumerate(sample_vectors):
            vector_data.append({
                "id": i,
                "vector": vector,
                "payload": {"text": f"Sample text {i}"}
            })
        
        result = clustering_engine.recluster_vectors(vector_data)
        
        # Check that each point has a cluster label
        for point in result["clustered_points"]:
            assert "cluster_id" in point
            assert isinstance(point["cluster_id"], int)
            assert point["cluster_id"] >= 0
    
    def test_noise_handling(self, clustering_engine, sample_vectors):
        """Test handling of noise points (cluster_id = -1)."""
        # Add some noise points
        noise_vectors = np.random.uniform(0, 15, (5, 3)).tolist()
        
        vector_data = []
        for i, vector in enumerate(sample_vectors + noise_vectors):
            vector_data.append({
                "id": i,
                "vector": vector,
                "payload": {"text": f"Sample text {i}"}
            })
        
        result = clustering_engine.recluster_vectors(vector_data)
        
        # Check that noise points are handled
        cluster_ids = [point["cluster_id"] for point in result["clustered_points"]]
        assert -1 in cluster_ids  # Noise points should have cluster_id = -1 