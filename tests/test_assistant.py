import os
import sqlite3
import tempfile
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from config import settings
from core.event_engine import EventEngine
from core.rag_memory import SurveillanceMemoryManager
from core.assistant import InvestigationAssistant

@pytest.fixture
def temp_db_and_index():
    """Fixture that initializes a temporary SQLite DB and FAISS index path."""
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    index_fd, index_path = tempfile.mkstemp(suffix=".faiss")
    
    # Store original values
    orig_db = settings.DB_PATH
    orig_index = settings.RAG_INDEX_PATH
    orig_key = settings.GEMINI_API_KEY
    
    settings.DB_PATH = Path(db_path)
    settings.RAG_INDEX_PATH = Path(index_path)
    settings.GEMINI_API_KEY = None
    
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
    settings.GEMINI_API_KEY = orig_key

def test_offline_summarizer(temp_db_and_index):
    """Verifies that the offline summarizer creates a formatted, structured chronological report."""
    engine = temp_db_and_index
    manager = SurveillanceMemoryManager(engine)
    assistant = InvestigationAssistant(manager, engine)
    
    # 1. Setup camera and target data
    t1 = "2026-06-27T10:00:00Z"
    t2 = "2026-06-27T10:05:00Z"
    with engine._get_connection() as conn:
        conn.execute(f"INSERT INTO persons (person_id, name, category, risk_level, created_at) VALUES ('P001', 'John Doe', 'Watchlist', 'High', '{t1}');")
        conn.execute("INSERT INTO cameras (camera_id, location, video_source) VALUES ('CAM_GATE', 'Front Entrance Gate', 'url');")
        conn.execute("INSERT INTO cameras (camera_id, location, video_source) VALUES ('CAM_LOBBY', 'Main Lobby', 'url');")
        conn.execute(f"INSERT INTO events (event_id, person_id, camera_id, video_source, timestamp, frame_number, confidence, bounding_box, match_details) VALUES ('EVT_1', 'P001', 'CAM_GATE', 'url', '{t1}', 10, 0.85, '[]', '{{}}');")
        conn.execute(f"INSERT INTO alerts (alert_id, event_id, status, severity_score, evidence_path, created_at, updated_at) VALUES ('ALT_1', 'EVT_1', 'ACTIVE', 0.85, 'C:\\evidence\\crop.jpg', '{t1}', '{t1}');")
        conn.commit()

    # 2. Mock RAG memory search results
    mock_sources = [
        {
            "memory_id": "MEM_001",
            "reference_id": "EVT_1",
            "entity_type": "event",
            "person_id": "P001",
            "camera_id": "CAM_GATE",
            "timestamp": t1,
            "document_text": "Sighting Event: John Doe at Front Entrance Gate",
            "evidence_path": "C:\\evidence\\crop.jpg",
            "similarity": 0.88
        },
        {
            "memory_id": "MEM_002",
            "reference_id": "ALT_1",
            "entity_type": "alert",
            "person_id": "P001",
            "camera_id": "CAM_GATE",
            "timestamp": t1,
            "document_text": "Alert: Active watchlist alert for John Doe at Front Entrance Gate",
            "evidence_path": "C:\\evidence\\crop.jpg",
            "similarity": 0.82
        },
        {
            "memory_id": "MEM_003",
            "reference_id": "MVT_1",
            "entity_type": "movement",
            "person_id": "P001",
            "camera_id": "CAM_LOBBY",
            "timestamp": t2,
            "document_text": "Movement Transition: John Doe moved from Front Entrance Gate to Main Lobby",
            "evidence_path": None,
            "similarity": 0.75
        }
    ]
    
    summary = assistant._generate_offline_summary("Where was John Doe?", mock_sources)
    
    # 3. Assert report structure is correct
    assert "### Executive Summary" in summary
    assert "### Chronological Timeline" in summary
    assert "### Cameras Involved" in summary
    assert "### Persons Involved" in summary
    assert "### Alerts Triggered" in summary
    assert "### Evidence References" in summary
    assert "### Confidence / Source Count" in summary
    
    # Check details
    assert "Front Entrance Gate" in summary
    assert "Main Lobby" in summary
    assert "John Doe" in summary
    assert "P001" in summary
    assert "ALT_1" in summary
    assert "crop.jpg" in summary
    assert "Average Match Confidence" in summary

def test_audit_logging(temp_db_and_index):
    """Verifies that query audit logging writes query sessions to assistant_queries_audit table."""
    engine = temp_db_and_index
    manager = SurveillanceMemoryManager(engine)
    assistant = InvestigationAssistant(manager, engine)
    
    # Add dummy memory
    t1 = datetime.utcnow().isoformat()
    with engine._get_connection() as conn:
        conn.execute(f"INSERT INTO persons (person_id, name, category, risk_level, created_at) VALUES ('P001', 'Alice', 'VIP', 'Low', '{t1}');")
        conn.execute("INSERT INTO cameras (camera_id, location, video_source) VALUES ('CAM_A', 'Lobby', 'url');")
        conn.commit()
        
    manager.add_memory("EVT_1", "event", "P001", "CAM_A", t1, "Alice was seen in Lobby")
    
    # Query assistant
    response, sources, backend = assistant.generate_response(
        user_query="Who was in the lobby?",
        operator_username="test_operator"
    )
    
    # Query database audit table
    with engine._get_connection() as conn:
        cursor = conn.execute("SELECT * FROM assistant_queries_audit WHERE user_username = 'test_operator';")
        rows = cursor.fetchall()
        assert len(rows) == 1
        audit = dict(rows[0])
        assert audit["query"] == "Who was in the lobby?"
        assert audit["backend_used"] == "Offline"
        assert audit["response_latency_ms"] >= 0
        retrieved_ids = json.loads(audit["retrieved_memory_ids"])
        assert len(retrieved_ids) >= 1

@patch("requests.post")
def test_gemini_backend_switching(mock_post, temp_db_and_index):
    """Verifies backend auto-upgrades to Gemini and falls back to Offline summarizer on failure."""
    engine = temp_db_and_index
    manager = SurveillanceMemoryManager(engine)
    assistant = InvestigationAssistant(manager, engine)
    
    # Setup key
    assistant.api_key = "MOCK_GEMINI_KEY"
    
    # Setup mock successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [{
            "content": {
                "parts": [{"text": "Gemini generated answer based on RAG context."}]
            }
        }]
    }
    mock_post.return_value = mock_response
    
    # 1. Test successful Gemini query
    res_text, sources, backend = assistant.generate_response("Where is John Doe?", operator_username="admin")
    assert backend == "Gemini"
    assert res_text == "Gemini generated answer based on RAG context."
    
    # 2. Test fallback on Gemini HTTP error status
    mock_response.status_code = 403
    res_text2, sources2, backend2 = assistant.generate_response("Where is John Doe?", operator_username="admin")
    assert backend2 == "Offline"
    assert "### Executive Summary" in res_text2  # Offline summarizer output
