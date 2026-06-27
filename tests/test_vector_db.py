import sys
import tempfile
import sqlite3
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.event_engine import EventEngine
from core.vector_db import VectorDBManager

def test_vector_db_operations():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_visionguard.db"
        index_path = Path(tmpdir) / "test_index.faiss"
        dim = 128

        # 1. Initialize EventEngine and VectorDBManager
        event_engine = EventEngine(db_path)
        vector_db = VectorDBManager(index_path, dim, event_engine)

        assert index_path.exists()
        assert vector_db.ntotal == 0

        # Enroll targets
        # Target 1 (John Doe)
        event_engine.register_person("T1", "John Doe", "Watchlist", "High")
        emb1 = np.random.rand(dim).astype('float32')
        # Normalize
        emb1 = emb1 / np.linalg.norm(emb1)
        
        emb1_id = vector_db.add_face(
            person_id="T1",
            embedding=emb1,
            image_path="t1.jpg",
            source_type="manual_enrollment",
            metadata={"notes": "Target 1 main profile"}
        )
        assert emb1_id.startswith("EMB_")
        assert vector_db.ntotal == 1

        # Enroll Target 1 second embedding (e.g. side profile)
        emb1_alt = emb1 + 0.05 * np.random.rand(dim).astype('float32')
        emb1_alt = emb1_alt / np.linalg.norm(emb1_alt)
        
        emb1_alt_id = vector_db.add_face(
            person_id="T1",
            embedding=emb1_alt,
            image_path="t1_alt.jpg",
            source_type="manual_enrollment",
            metadata={"notes": "Target 1 side profile"}
        )
        assert vector_db.ntotal == 2

        # Target 2 (Jane Smith)
        event_engine.register_person("T2", "Jane Smith", "VIP", "Low")
        # Ensure emb2 is orthogonal/distinct
        emb2 = np.random.rand(dim).astype('float32')
        emb2 = emb2 - np.dot(emb2, emb1) * emb1 # Make orthogonal
        emb2 = emb2 / np.linalg.norm(emb2)

        emb2_id = vector_db.add_face(
            person_id="T2",
            embedding=emb2,
            image_path="t2.jpg",
            source_type="manual_enrollment",
            metadata={"notes": "Target 2 front"}
        )
        assert vector_db.ntotal == 3

        # Save and Reload Persistence Test
        vector_db.save()
        
        # Load in a new VectorDBManager instance
        new_vector_db = VectorDBManager(index_path, dim, event_engine)
        assert new_vector_db.ntotal == 3

        # 2. SQLite Mapping Integrity
        person1 = event_engine.get_person("T1")
        assert person1 is not None
        assert person1["name"] == "John Doe"
        assert len(person1["embeddings"]) == 2
        assert person1["embeddings"][0]["vector_id"] == 0
        assert person1["embeddings"][1]["vector_id"] == 1

        person2 = event_engine.get_person("T2")
        assert person2 is not None
        assert len(person2["embeddings"]) == 1
        assert person2["embeddings"][0]["vector_id"] == 2

        # 3. Accurate Top-K Retrieval
        # Query with something close to emb1
        query_vector = emb1 + 0.01 * np.random.rand(dim).astype('float32')
        query_vector = query_vector / np.linalg.norm(query_vector)

        matches = new_vector_db.search(query_vector, k=2, threshold=0.70)
        assert len(matches) > 0
        top_match = matches[0]
        assert top_match["person_id"] == "T1"
        assert top_match["name"] == "John Doe"
        assert top_match["similarity"] > 0.90

        # Query with something close to emb2
        query_vector_2 = emb2 + 0.01 * np.random.rand(dim).astype('float32')
        query_vector_2 = query_vector_2 / np.linalg.norm(query_vector_2)
        matches_2 = new_vector_db.search(query_vector_2, k=1, threshold=0.70)
        assert len(matches_2) == 1
        assert matches_2[0]["person_id"] == "T2"

        # 4. Search Audit Logging
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM search_audit_logs;")
        audit_count = cursor.fetchone()[0]
        # Should have 2 audit logs (one for each search call)
        assert audit_count == 2
        
        cursor = conn.execute("SELECT query_source, results_count FROM search_audit_logs;")
        logs = cursor.fetchall()
        assert logs[0][0] == "manual_query"
        conn.close()

def test_configurable_dimensions():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_visionguard_512.db"
        index_path = Path(tmpdir) / "test_index_512.faiss"
        dim_512 = 512

        event_engine = EventEngine(db_path)
        vector_db = VectorDBManager(index_path, dim_512, event_engine)

        assert vector_db.dimension == 512
        assert vector_db.index.d == 512

        event_engine.register_person("T_512", "ArcFace User", "Watchlist", "Critical")
        emb_512 = np.random.rand(dim_512).astype('float32')
        emb_512 = emb_512 / np.linalg.norm(emb_512)

        emb_id = vector_db.add_face(
            person_id="T_512",
            embedding=emb_512,
            image_path="t_512.jpg",
            source_type="manual_enrollment",
            metadata={"notes": "512-D test vector"}
        )
        assert vector_db.ntotal == 1
