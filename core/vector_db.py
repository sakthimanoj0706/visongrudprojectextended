import faiss
import numpy as np
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from core.event_engine import EventEngine

class VectorDBManager:
    def __init__(self, index_path: Path, dimension: int, event_engine: EventEngine):
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.event_engine = event_engine
        
        # Load or initialize the FAISS Index
        if self.index_path.exists() and self.index_path.stat().st_size > 0:
            print(f"[VECTOR DB] Loading existing FAISS index from {self.index_path}...")
            self.index = faiss.read_index(str(self.index_path))
            # Verify dimension matches loaded index
            if self.index.d != self.dimension:
                raise ValueError(f"FAISS index dimension mismatch. Expected {self.dimension}, got {self.index.d}")
        else:
            print(f"[VECTOR DB] Initializing new FAISS IndexFlatIP (Dimension: {self.dimension})...")
            # IndexFlatIP uses Inner Product which corresponds to Cosine Similarity when vectors are L2 normalized
            self.index = faiss.IndexFlatIP(self.dimension)
            self.save()

    def add_face(self, person_id: str, embedding: np.ndarray, image_path: str,
                 source_type: str, metadata: Dict[str, Any]) -> str:
        """
        Adds a face embedding vector to the FAISS index and registers the mapping in SQLite.
        
        Args:
            person_id (str): The ID of the person this embedding belongs to.
            embedding (np.ndarray): 1D or 2D array representation of the embedding.
            image_path (str): Path to the cropped face image or full enrollment frame.
            source_type (str): Origin of embedding ("manual_enrollment", "cctv_sighting").
            metadata (Dict[str, Any]): Additional info (crop box coordinates, confidence, etc.).
            
        Returns:
            str: Generated unique embedding_id.
        """
        # Format embedding to float32 2D row vector
        emb_arr = np.atleast_2d(embedding).astype('float32')
        if emb_arr.shape[1] != self.dimension:
            raise ValueError(f"Embedding size {emb_arr.shape[1]} does not match database dimension {self.dimension}")

        # Ensure embedding is L2-normalized
        norm = np.linalg.norm(emb_arr)
        if norm > 0:
            emb_arr = emb_arr / norm

        # Add to FAISS index
        self.index.add(emb_arr)
        vector_id = self.index.ntotal - 1  # FAISS adds sequentially, index is ntotal-1

        # Generate unique ID and insert mapping in SQLite database
        embedding_id = f"EMB_{uuid.uuid4().hex[:8].upper()}"
        success = self.event_engine.register_face_embedding(
            embedding_id=embedding_id,
            person_id=person_id,
            vector_id=vector_id,
            image_path=image_path,
            source_type=source_type,
            metadata=metadata
        )
        if not success:
            # Rollback index add if SQL registration fails by removing the added vector (not supported in IndexFlat easily,
            # but we throw an error to alert the application layer).
            raise RuntimeError(f"Failed to record mapping in database for vector_id {vector_id}")

        return embedding_id

    def search(self, embedding: np.ndarray, k: int = 5, threshold: float = 0.40,
               query_source: str = "manual_query") -> List[Dict[str, Any]]:
        """
        Queries FAISS for the Top-K nearest face embeddings, evaluates threshold,
        maps results to person metadata, and records audit logs.
        
        Returns:
            List[Dict[str, Any]]: List of matching face objects containing similarity, person, and metadata.
        """
        start_time = time.perf_counter()
        
        # Prepare embedding
        emb_arr = np.atleast_2d(embedding).astype('float32')
        if emb_arr.shape[1] != self.dimension:
            raise ValueError(f"Query embedding size {emb_arr.shape[1]} does not match database dimension {self.dimension}")
            
        # Normalize query vector
        norm = np.linalg.norm(emb_arr)
        if norm > 0:
            emb_arr = emb_arr / norm

        # Execute FAISS vector search
        # FAISS search returns: (distances: float[][], indices: int[][])
        # Since it is IndexFlatIP, distances represent raw Inner Product (cosine similarity)
        k = min(k, self.index.ntotal)
        if k == 0:
            # Index is empty
            self._log_audit(query_source, k, threshold, 0, start_time)
            return []

        distances, indices = self.index.search(emb_arr, k)
        
        matches = []
        raw_distances = distances[0]
        raw_indices = indices[0]

        for dist, vec_id in zip(raw_distances, raw_indices):
            # FAISS returns index -1 if not enough matches are found
            if vec_id == -1:
                continue
                
            similarity = float(dist)
            
            # Filter matches by similarity threshold
            if similarity >= threshold:
                # Query mapping
                person_info = self.event_engine.get_person_by_vector_id(int(vec_id))
                if person_info:
                    matches.append({
                        "person_id": person_info["person_id"],
                        "name": person_info["name"],
                        "category": person_info["category"],
                        "risk_level": person_info["risk_level"],
                        "similarity": similarity,
                        "embedding_id": person_info["embedding_id"],
                        "image_path": person_info["image_path"],
                        "source_type": person_info["source_type"],
                        "emb_metadata": person_info["emb_metadata"]
                    })

        self._log_audit(query_source, k, threshold, len(matches), start_time)
        return matches

    def _log_audit(self, source: str, k: int, threshold: float, count: int, start_time: float):
        """Logs vector search transactions to database audit trail."""
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        log_id = f"AUD_{uuid.uuid4().hex[:8].upper()}"
        params = {
            "top_k": k,
            "threshold": threshold,
            "index_size": self.index.ntotal
        }
        self.event_engine.log_search_audit(
            log_id=log_id,
            query_source=source,
            query_params=params,
            results_count=count,
            execution_time_ms=duration_ms
        )

    def save(self):
        """Persists the FAISS index file to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        print(f"[VECTOR DB] Saved FAISS index with {self.index.ntotal} vectors to {self.index_path}")

    @property
    def ntotal(self) -> int:
        return self.index.ntotal
