"""Clustering engine for 42."""

import numpy as np
from typing import List, Dict, Optional
from loguru import logger
import matplotlib.pyplot as plt
from pathlib import Path

from .interfaces import Chunk, Cluster
from .vector_store import VectorStore


class ClusteringEngine:
    """Clusters similar code patterns using HDBSCAN and UMAP."""
    
    def __init__(self):
        """Initialize the clustering engine."""
        self.vector_store = VectorStore()
        
    def recluster_vectors(self, min_cluster_size: int = 5, min_samples: int = 3) -> Dict[int, Cluster]:
        """Recluster all vectors using HDBSCAN and update payloads with cluster IDs."""
        try:
            # Import here to avoid dependency issues
            import hdbscan
            import umap
            
            logger.info("Starting vector reclustering...")
            
            # Get all vectors from the store
            all_vectors = self.vector_store.get_all_vectors()
            
            if not all_vectors:
                logger.warning("No vectors found for clustering")
                return {}
            
            # Extract vectors and metadata
            vectors = []
            metadata_list = []
            for vector_data in all_vectors:
                vectors.append(vector_data["vector"])
                metadata_list.append(vector_data.get("metadata", {}))
            
            vectors = np.array(vectors)
            
            # Reduce dimensionality with UMAP
            logger.info("Reducing dimensionality with UMAP...")
            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                n_components=2,
                random_state=42
            )
            reduced_vectors = reducer.fit_transform(vectors)
            
            # Cluster with HDBSCAN
            logger.info("Clustering with HDBSCAN...")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean'
            )
            cluster_labels = clusterer.fit_predict(reduced_vectors)
            
            # Group chunks by cluster
            clusters = {}
            for i, (vector_data, label) in enumerate(zip(all_vectors, cluster_labels)):
                if label == -1:  # Noise points
                    continue
                
                # Create chunk object with safe field access
                chunk = Chunk(
                    text=vector_data.get("text", ""),
                    file_path=vector_data.get("file_path", ""),
                    start_line=vector_data.get("start_line", 0),
                    end_line=vector_data.get("end_line", 0),
                    cluster_id=label,
                    metadata=vector_data.get("metadata", {})
                )
                
                # Add to cluster
                if label not in clusters:
                    clusters[label] = Cluster(
                        cluster_id=label,
                        chunks=[],
                        centroid=vector_data["vector"],
                        size=0
                    )
                
                clusters[label].chunks.append(chunk)
                clusters[label].size += 1
                
                # Update centroid (simple average)
                if clusters[label].centroid is None:
                    clusters[label].centroid = vector_data["vector"]
                else:
                    clusters[label].centroid = np.mean([
                        clusters[label].centroid,
                        vector_data["vector"]
                    ], axis=0)
            
            # Update vector store with cluster IDs
            logger.info(f"Updating {len(all_vectors)} vectors with cluster IDs...")
            for i, (vector_data, label) in enumerate(zip(all_vectors, cluster_labels)):
                # Update payload with cluster ID
                updated_payload = {k: v for k, v in vector_data.items() if k != 'vector'}  # Remove vector from payload
                updated_payload["cluster_id"] = int(label)
                
                # Get the actual point ID from the vector data
                point_id = vector_data.get('id', i)
                
                # Update in vector store
                self.vector_store.update_payload(
                    point_id=point_id,  # Use actual point ID
                    payload=updated_payload
                )
            
            # Generate cluster visualization
            self._generate_cluster_plot(reduced_vectors, cluster_labels)
            
            logger.info(f"Clustering complete: {len(clusters)} clusters found")
            return clusters
            
        except ImportError as e:
            logger.error(f"Missing clustering dependencies: {e}")
            logger.info("Install with: pip install hdbscan umap-learn matplotlib")
            return {}
        except Exception as e:
            logger.error(f"Failed to recluster vectors: {e}")
            return {}
    
    def _generate_cluster_plot(self, reduced_vectors: np.ndarray, cluster_labels: np.ndarray) -> None:
        """Generate a UMAP visualization of clusters."""
        try:
            # Create docs directory if it doesn't exist
            docs_dir = Path("docs")
            docs_dir.mkdir(exist_ok=True)
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Plot points by cluster
            unique_labels = set(cluster_labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = cluster_labels == label
                if label == -1:
                    # Noise points in black
                    plt.scatter(
                        reduced_vectors[mask, 0],
                        reduced_vectors[mask, 1],
                        c='black',
                        s=20,
                        alpha=0.5,
                        label=f'Noise ({np.sum(mask)} points)'
                    )
                else:
                    plt.scatter(
                        reduced_vectors[mask, 0],
                        reduced_vectors[mask, 1],
                        c=[color],
                        s=30,
                        alpha=0.7,
                        label=f'Cluster {label} ({np.sum(mask)} points)'
                    )
            
            plt.title('Code Pattern Clusters (UMAP + HDBSCAN)')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save the plot
            plot_path = docs_dir / "cluster.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Cluster visualization saved to {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate cluster plot: {e}")
    
    def get_cluster_stats(self) -> Dict:
        """Get statistics about current clusters."""
        try:
            all_vectors = self.vector_store.get_all_vectors()
            
            cluster_stats = {}
            total_vectors = len(all_vectors)
            
            for vector_data in all_vectors:
                cluster_id = vector_data.get("cluster_id", -1)
                if cluster_id not in cluster_stats:
                    cluster_stats[cluster_id] = {
                        "size": 0,
                        "files": set(),
                        "avg_score": 0.0
                    }
                
                cluster_stats[cluster_id]["size"] += 1
                cluster_stats[cluster_id]["files"].add(vector_data.get("file_path", "unknown"))
            
            # Convert sets to counts and calculate averages
            for cluster_id, stats in cluster_stats.items():
                stats["files"] = len(stats["files"])
                stats["percentage"] = (stats["size"] / total_vectors) * 100 if total_vectors > 0 else 0
            
            return {
                "total_vectors": total_vectors,
                "total_clusters": len([c for c in cluster_stats.keys() if c != -1]),
                "noise_points": cluster_stats.get(-1, {"size": 0})["size"],
                "clusters": cluster_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get cluster stats: {e}")
            return {} 