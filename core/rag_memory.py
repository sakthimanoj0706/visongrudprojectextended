import os
import re
import json
import uuid
import threading
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from config import settings
from core.event_engine import EventEngine

# --- Pluggable Text Embedder Interface ---

class BaseTextEmbedder:
    def embed(self, text: str) -> np.ndarray:
        """Returns a 384-dimensional normalized float embedding vector."""
        raise NotImplementedError

# --- Default: Deterministic Semantic Text Vectorizer (DSTV) ---

class DSTVTextEmbedder(BaseTextEmbedder):
    """
    Lightweight, CPU-friendly, dependency-free text embedder.
    Generates a 384-D normalized vector based on a surveillance-specific vocabulary.
    Includes custom word importance weights (boosting key entities, locations, and actions).
    """
    def __init__(self):
        # 1. Define core surveillance terms
        self.core_words = [
            "sighting", "event", "tracklet", "movement", "alert", "rule", "operator", "evidence",
            "active", "acknowledged", "resolved", "false_positive", "online", "offline", "reconnecting", "degraded",
            "seen", "sighted", "located", "detected", "tracked", "transitioned", "moved", "triggered", "notified", "sent", "archived",
            "gate", "lobby", "entrance", "exit", "elevator", "office", "building", "cctv", "camera", "front", "main", "side",
            "low", "medium", "high", "critical",
            "person", "id", "name", "target", "watchlist", "unknown", "suspect", "vip", "restricted",
            "morning", "afternoon", "evening", "night", "day", "today", "yesterday", "time", "hour", "minute", "second", "duration",
            "similarity", "score", "confidence", "threshold", "match", "aligned", "crop", "frame",
            "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
            "p001", "target_001", "trk", "alt", "evt", "cam", "mvt", "mem", "notes", "investigation"
        ]
        
        # 2. Pad vocabulary deterministically to exactly 384 dimensions to align with standard RAG models
        self.vocabulary = list(self.core_words)
        pad_idx = 0
        while len(self.vocabulary) < 384:
            self.vocabulary.append(f"pad_{pad_idx}")
            pad_idx += 1
            
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}

    def embed(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(384, dtype=np.float32)

        # Tokenize (lowercase, strip non-alphanumeric characters)
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        vector = np.zeros(384, dtype=np.float32)
        for token in tokens:
            if token in self.word_to_idx:
                idx = self.word_to_idx[token]
                weight = 1.0
                
                # Boost key entity matches to ensure high retrieval priority
                if token.startswith("target_") or token.startswith("unknown_") or token == "p001":
                    weight = 5.0
                elif token.startswith("cam_") or token in ["gate", "lobby", "office", "elevator"]:
                    weight = 3.0
                elif token in ["critical", "high", "alert", "watchlist", "sighting"]:
                    weight = 2.0
                    
                vector[idx] += weight

        # L2 Normalize the vector to enable cosine similarity via Inner Product
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

# --- Optional: Deep Learning SentenceTransformers Embedder ---

class SentenceTransformersTextEmbedder(BaseTextEmbedder):
    """Loads MiniLM or similar text embedder model using SentenceTransformers library."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            print("[RAG WARNING] sentence-transformers package not installed. Falling back to DSTV.")
            self.model = None

    def embed(self, text: str) -> np.ndarray:
        if self.model is None:
            # Fallback to DSTV
            return DSTVTextEmbedder().embed(text)
        vector = self.model.encode(text, convert_to_numpy=True)
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

# --- Memory Document Text Templates ---

def formulate_sighting_text(name: str, person_id: str, camera_location: str, camera_id: str, timestamp: str, similarity: float) -> str:
    return f"Sighting Event: Watchlist target {name} (person ID: {person_id}) was sighted at {camera_location} (camera ID: {camera_id}) on {timestamp} with a face match similarity of {similarity:.2f}."

def formulate_alert_text(name: str, person_id: str, camera_location: str, camera_id: str, timestamp: str, severity: float, status: str) -> str:
    return f"Surveillance Alert: An operational alert was triggered for target {name} (person ID: {person_id}) at {camera_location} (camera ID: {camera_id}) on {timestamp}. Severity score: {severity:.2f}. Current status is {status}."

def formulate_tracklet_text(name: str, person_id: str, camera_location: str, camera_id: str, start_time: str, end_time: Optional[str], status: str) -> str:
    end_time_str = end_time if end_time else "still active"
    return f"Tracklet Session: Watchlist target {name} (person ID: {person_id}) was tracked on camera {camera_id} ({camera_location}) starting at {start_time} and ending at {end_time_str}. Tracklet session status is {status}."

def formulate_movement_text(name: str, person_id: str, from_loc: str, from_cam: str, to_loc: str, to_cam: str, departure_time: str, arrival_time: str, duration: float, similarity: float) -> str:
    return f"Movement Transition: Target {name} (person ID: {person_id}) moved from {from_loc} ({from_cam}) to {to_loc} ({to_cam}). Departed at {departure_time}, arrived at {arrival_time}. Transit duration took {duration:.1f} seconds with a ReID similarity of {similarity:.2f}."

# --- FAISS RAG Memory Index Manager ---

_rag_lock = threading.Lock()

class SurveillanceMemoryManager:
    def __init__(self, event_engine: EventEngine):
        self.event_engine = event_engine
        
        # 1. Instantiate the pluggable embedder based on settings
        if settings.TEXT_EMBEDDING_MODE == "sentence-transformers":
            self.embedder = SentenceTransformersTextEmbedder()
        else:
            self.embedder = DSTVTextEmbedder()

        self._init_index()

    def _init_index(self):
        """Initializes or loads the FAISS RAG memory index."""
        settings.RAG_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        if settings.RAG_INDEX_PATH.exists():
            try:
                self.index = faiss.read_index(str(settings.RAG_INDEX_PATH))
                return
            except Exception as e:
                print(f"[RAG WARNING] Failed to load index: {e}. Re-initializing index.")
                
        # Initialize a new Inner Product (Cosine Similarity) index
        self.index = faiss.IndexFlatIP(settings.RAG_EMBEDDING_DIM)
        print("[RAG MEMORY] Created a new RAG text memory FAISS index.")

    def save_index(self):
        """Persists the FAISS text index to disk."""
        try:
            faiss.write_index(self.index, str(settings.RAG_INDEX_PATH))
        except Exception as e:
            print(f"[RAG ERROR] Failed to save FAISS text index: {e}")

    def _add_memory_internal(self, reference_id: str, entity_type: str, person_id: Optional[str],
                             camera_id: Optional[str], timestamp: str, document_text: str,
                             evidence_path: Optional[str] = None) -> str:
        """Internal, un-locked memory logging method."""
        self.event_engine.delete_memory_by_reference(reference_id)
        vector = self.embedder.embed(document_text)
        self.index.add(np.array([vector]))
        vector_id = self.index.ntotal - 1
        
        memory_id = f"MEM_{uuid.uuid4().hex[:8].upper()}"
        success = self.event_engine.log_memory_document(
            memory_id=memory_id,
            reference_id=reference_id,
            entity_type=entity_type,
            person_id=person_id,
            camera_id=camera_id,
            timestamp=timestamp,
            document_text=document_text,
            vector_id=vector_id,
            evidence_path=evidence_path
        )
        if success:
            self.save_index()
            return memory_id
        return ""

    def add_memory(self, reference_id: str, entity_type: str, person_id: Optional[str],
                   camera_id: Optional[str], timestamp: str, document_text: str,
                   evidence_path: Optional[str] = None) -> str:
        """Embeds and logs a natural language memory document into the database and FAISS index."""
        with _rag_lock:
            self._init_index()  # Reload latest persisted index to capture concurrent edits
            return self._add_memory_internal(
                reference_id=reference_id,
                entity_type=entity_type,
                person_id=person_id,
                camera_id=camera_id,
                timestamp=timestamp,
                document_text=document_text,
                evidence_path=evidence_path
            )

    def search_memory(self, query_text: str, k: int = 5,
                      person_id: Optional[str] = None,
                      camera_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Queries RAG memory using vector similarity.
        Applies metadata filtering before returning results.
        """
        with _rag_lock:
            self._init_index()  # Reload latest index to search up-to-date data
            
            if self.index.ntotal == 0:
                return []

            # 1. Embed query
            query_vector = self.embedder.embed(query_text)
            
            # 2. Query FAISS index (fetch more candidates to allow metadata filtering)
            search_k = min(self.index.ntotal, k * 5)
            distances, indices = self.index.search(np.array([query_vector]), search_k)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                    
                # Map FAISS vector_id back to database document
                memory_row = self.event_engine.get_memory_by_vector_id(int(idx))
                if not memory_row:
                    continue
                    
                # Apply metadata filters
                if person_id and memory_row["person_id"] != person_id:
                    continue
                if camera_id and memory_row["camera_id"] != camera_id:
                    continue
                    
                results.append({
                    "memory_id": memory_row["memory_id"],
                    "reference_id": memory_row["reference_id"],
                    "entity_type": memory_row["entity_type"],
                    "person_id": memory_row["person_id"],
                    "camera_id": memory_row["camera_id"],
                    "timestamp": memory_row["timestamp"],
                    "document_text": memory_row["document_text"],
                    "evidence_path": memory_row["evidence_path"],
                    "similarity": float(dist)
                })
                
                if len(results) >= k:
                    break
                    
            return results

    def rebuild_memory(self) -> int:
        """
        Deletes the FAISS index and surveillance_memory database table,
        and regenerates RAG memory records from scratch.
        """
        with _rag_lock:
            print("[RAG MEMORY] Rebuilding RAG surveillance memory index...")
            
            # 1. Clear database surveillance memory table
            try:
                with self.event_engine._get_connection() as conn:
                    conn.execute("DELETE FROM surveillance_memory;")
                    conn.commit()
            except Exception as e:
                print(f"[RAG ERROR] Failed to clear memory table: {e}")
                
            # 2. Reset FAISS index
            self.index = faiss.IndexFlatIP(settings.RAG_EMBEDDING_DIM)
            
            count = 0
            try:
                with self.event_engine._get_connection() as conn:
                    # A. Rebuild Sighting Events Memories
                    cursor = conn.execute("""
                        SELECT e.event_id, e.person_id, e.camera_id, e.timestamp, e.confidence, p.name, c.location
                        FROM events e
                        JOIN persons p ON e.person_id = p.person_id
                        JOIN cameras c ON e.camera_id = c.camera_id
                    """)
                    for row in cursor.fetchall():
                        text = formulate_sighting_text(row[5], row[1], row[6], row[2], row[3], row[4])
                        self._add_memory_internal(row[0], "event", row[1], row[2], row[3], text, None)
                        count += 1

                    # B. Rebuild Operational Alerts Memories
                    cursor = conn.execute("""
                        SELECT a.alert_id, e.person_id, e.camera_id, a.created_at, a.severity_score, a.status, p.name, c.location
                        FROM alerts a
                        JOIN events e ON a.event_id = e.event_id
                        JOIN persons p ON e.person_id = p.person_id
                        JOIN cameras c ON e.camera_id = c.camera_id
                    """)
                    for row in cursor.fetchall():
                        text = formulate_alert_text(row[6], row[1], row[7], row[2], row[3], row[4], row[5])
                        self._add_memory_internal(row[0], "alert", row[1], row[2], row[3], text, None)
                        count += 1

                    # C. Rebuild Tracklet Memories
                    cursor = conn.execute("""
                        SELECT t.tracklet_id, t.person_id, t.camera_id, t.start_time, t.end_time, t.status, p.name, c.location
                        FROM tracklets t
                        LEFT JOIN persons p ON t.person_id = p.person_id
                        LEFT JOIN cameras c ON t.camera_id = c.camera_id
                    """)
                    for row in cursor.fetchall():
                        name = row[6] if row[6] else "Unknown Target"
                        loc = row[7] if row[7] else "Unknown Location"
                        text = formulate_tracklet_text(name, row[1] or "unregistered", loc, row[2], row[3], row[4], row[5])
                        self._add_memory_internal(row[0], "tracklet", row[1], row[2], row[3], text, None)
                        count += 1

                    # D. Rebuild Movements Memories
                    cursor = conn.execute("""
                        SELECT m.movement_id, m.person_id, m.from_camera_id, m.to_camera_id, m.departure_time, m.arrival_time, m.duration_seconds, m.similarity, p.name, c1.location, c2.location
                        FROM camera_movements m
                        JOIN persons p ON m.person_id = p.person_id
                        JOIN cameras c1 ON m.from_camera_id = c1.camera_id
                        JOIN cameras c2 ON m.to_camera_id = c2.camera_id
                    """)
                    for row in cursor.fetchall():
                        text = formulate_movement_text(row[8], row[1], row[9], row[2], row[10], row[3], row[4], row[5], row[6], row[7])
                        self._add_memory_internal(row[0], "movement", row[1], row[3], row[5], text, None)
                        count += 1

            except Exception as e:
                print(f"[RAG ERROR] Rebuild failed: {e}")
                
            print(f"[RAG MEMORY] Rebuild completed. Indexed {count} memory documents.")
            return count

