import os
import sqlite3
import tempfile
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime

from config import settings
from core.event_engine import EventEngine
from core.rag_memory import (
    SurveillanceMemoryManager,
    DSTVTextEmbedder,
    formulate_sighting_text,
    formulate_alert_text,
    formulate_tracklet_text,
    formulate_movement_text
)

@pytest.fixture
def temp_db_and_index():
    """Fixture that initializes a temporary SQLite DB and FAISS index path."""
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    index_fd, index_path = tempfile.mkstemp(suffix=".faiss")
    
    # Store original values
    orig_db = settings.DB_PATH
    orig_index = settings.RAG_INDEX_PATH
    
    settings.DB_PATH = Path(db_path)
    settings.RAG_INDEX_PATH = Path(index_path)
    
    # Initialize DB
    engine = EventEngine(settings.DB_PATH)
    
    yield engine
    
    # Cleanup files
    os.close(db_fd)
    os.close(index_fd)
    try:
        os.remove(db_path)
    except OSError:
        pass
    try:
        os.remove(index_path)
    except OSError:
        pass
        
    settings.DB_PATH = orig_db
    settings.RAG_INDEX_PATH = orig_index

def test_dstv_vectorizer():
    """Verifies that DSTV generates correct 384-D normalized vectors."""
    embedder = DSTVTextEmbedder()
    
    # 1. Dimension check
    vec1 = embedder.embed("Sighting at Front Gate CAM_GATE.")
    assert isinstance(vec1, np.ndarray)
    assert vec1.shape == (384,)
    
    # 2. Normalization check (L2 norm should be close to 1.0)
    norm = np.linalg.norm(vec1)
    assert pytest.approx(norm) == 1.0
    
    # 3. Empty text check
    vec_empty = embedder.embed("")
    assert np.all(vec_empty == 0.0)
    
    # 4. Keyword boosting check
    # An embed with 'critical' should have a higher similarity to a query with 'critical'
    vec_critical = embedder.embed("A critical alert was triggered.")
    vec_low = embedder.embed("A low priority sighting event occurred.")
    query_vec = embedder.embed("critical")
    
    sim_critical = float(np.dot(vec_critical, query_vec))
    sim_low = float(np.dot(vec_low, query_vec))
    assert sim_critical > sim_low

def test_memory_crud_and_faiss(temp_db_and_index):
    """Verifies adding, searching, and filtering memories in FAISS + SQLite."""
    engine = temp_db_and_index
    
    # Seed data to satisfy foreign key constraints
    t_now = datetime.utcnow().isoformat()
    with engine._get_connection() as conn:
        conn.execute(f"INSERT INTO persons (person_id, name, category, risk_level, created_at) VALUES ('P001', 'Alice Smith', 'Suspect', 'High', '{t_now}');")
        conn.execute(f"INSERT INTO persons (person_id, name, category, risk_level, created_at) VALUES ('P002', 'Bob Jones', 'VIP', 'Low', '{t_now}');")
        conn.execute("INSERT INTO cameras (camera_id, location, video_source) VALUES ('CAM_GATE', 'Front Gate', 'url');")
        conn.execute("INSERT INTO cameras (camera_id, location, video_source) VALUES ('CAM_LOBBY', 'Main Lobby', 'url');")
        conn.commit()

    manager = SurveillanceMemoryManager(engine)
    
    # Add dummy memories
    t1 = datetime.utcnow().isoformat()
    txt1 = formulate_sighting_text("Alice Smith", "P001", "Front Gate", "CAM_GATE", t1, 0.88)
    mem_id1 = manager.add_memory(
        reference_id="EVT_001",
        entity_type="event",
        person_id="P001",
        camera_id="CAM_GATE",
        timestamp=t1,
        document_text=txt1,
        evidence_path="/evidence/CAM_GATE/EVT_001.jpg"
    )
    assert mem_id1.startswith("MEM_")
    
    txt2 = formulate_alert_text("Bob Jones", "P002", "Main Lobby", "CAM_LOBBY", t1, 0.75, "ACTIVE")
    mem_id2 = manager.add_memory(
        reference_id="ALT_002",
        entity_type="alert",
        person_id="P002",
        camera_id="CAM_LOBBY",
        timestamp=t1,
        document_text=txt2
    )
    
    # Search memory
    results = manager.search_memory("Alice at Gate", k=5)
    assert len(results) >= 1
    # First match should be Alice's sighting
    assert results[0]["person_id"] == "P001"
    assert "Alice" in results[0]["document_text"]
    assert results[0]["evidence_path"] == "/evidence/CAM_GATE/EVT_001.jpg"
    
    # Test Metadata Filtering (Person ID)
    p_filtered = manager.search_memory("critical alert", k=5, person_id="P002")
    for r in p_filtered:
        assert r["person_id"] == "P002"
        
    # Test Metadata Filtering (Camera ID)
    c_filtered = manager.search_memory("sighting", k=5, camera_id="CAM_GATE")
    for r in c_filtered:
        assert r["camera_id"] == "CAM_GATE"

def test_memory_rebuild(temp_db_and_index):
    """Verifies that RAG memories can be accurately reconstructed from database tables."""
    engine = temp_db_and_index
    
    # 1. Setup seed data in base DB tables
    t_now = datetime.utcnow().isoformat()
    with engine._get_connection() as conn:
        # Register targets and cameras
        conn.execute(f"INSERT INTO persons (person_id, name, category, risk_level, created_at) VALUES ('P001', 'Charlie Brown', 'VIP', 'High', '{t_now}');")
        conn.execute("INSERT INTO cameras (camera_id, location, video_source) VALUES ('CAM_001', 'North Gate', 'stream_url');")
        
        # Log event
        conn.execute(f"INSERT INTO events (event_id, person_id, camera_id, video_source, timestamp, frame_number, confidence, bounding_box, match_details) VALUES ('EVT_REB', 'P001', 'CAM_001', 'url', '{t_now}', 10, 0.85, '[]', '{{}}');")
        
        # Log alert
        conn.execute(f"INSERT INTO alerts (alert_id, event_id, status, severity_score, evidence_path, created_at, updated_at) VALUES ('ALT_REB', 'EVT_REB', 'ACTIVE', 0.90, 'path', '{t_now}', '{t_now}');")
        
        # Log tracklet
        conn.execute(f"INSERT INTO tracklets (tracklet_id, camera_id, person_id, start_time, end_time, status) VALUES ('TRK_REB', 'CAM_001', 'P001', '{t_now}', '{t_now}', 'EXPIRED');")
        
        # Log camera movement
        conn.execute(f"INSERT INTO camera_movements (movement_id, person_id, from_camera_id, to_camera_id, departure_time, arrival_time, duration_seconds, similarity) VALUES ('MVT_REB', 'P001', 'CAM_001', 'CAM_001', '{t_now}', '{t_now}', 5.0, 0.95);")
        conn.commit()

    manager = SurveillanceMemoryManager(engine)
    
    # Trigger rebuild
    rebuild_count = manager.rebuild_memory()
    assert rebuild_count == 4  # 1 event + 1 alert + 1 tracklet + 1 movement
    
    # Search and verify
    results = manager.search_memory("Charlie Brown", k=10)
    assert len(results) == 4
    types = [r["entity_type"] for r in results]
    assert "event" in types
    assert "alert" in types
    assert "tracklet" in types
    assert "movement" in types

def test_alert_archive_sync(temp_db_and_index):
    """Verifies that RAG memory alerts are successfully deleted when alert archives are purged."""
    engine = temp_db_and_index
    manager = SurveillanceMemoryManager(engine)
    
    # Setup alert and memory
    t_old = "2020-01-01T00:00:00"  # Way older than 30 days retention
    with engine._get_connection() as conn:
        conn.execute(f"INSERT OR IGNORE INTO persons (person_id, name, category, risk_level, created_at) VALUES ('P100', 'Old Target', 'Suspect', 'High', '{t_old}');")
        conn.execute("INSERT OR IGNORE INTO cameras (camera_id, location, video_source) VALUES ('CAM_OLD', 'Old Camera', 'url');")
        conn.execute(f"INSERT INTO events (event_id, person_id, camera_id, video_source, timestamp, frame_number, confidence, bounding_box, match_details) VALUES ('EVT_OLD', 'P100', 'CAM_OLD', 'url', '{t_old}', 10, 0.85, '[]', '{{}}');")
        
        # Log alert with status FALSE_POSITIVE (eligible for retention purge)
        conn.execute(f"INSERT INTO alerts (alert_id, event_id, status, severity_score, evidence_path, created_at, updated_at) VALUES ('ALT_OLD', 'EVT_OLD', 'FALSE_POSITIVE', 0.50, 'path', '{t_old}', '{t_old}');")
        conn.commit()
        
    # Log alert memory
    txt = formulate_alert_text("Old Target", "P100", "Old Camera", "CAM_OLD", t_old, 0.50, "FALSE_POSITIVE")
    manager.add_memory("ALT_OLD", "alert", "P100", "CAM_OLD", t_old, txt)
    
    # Verify memory exists
    res = manager.search_memory("Old Target", k=1)
    assert len(res) == 1
    assert res[0]["reference_id"] == "ALT_OLD"
    
    # Run archiving purge
    archived = engine.archive_old_alerts(retention_days=30)
    assert archived == 1
    
    # Verify memory got deleted
    res2 = manager.search_memory("Old Target", k=1)
    assert len(res2) == 0
