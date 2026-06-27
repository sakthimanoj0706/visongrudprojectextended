import sys
import tempfile
import sqlite3
import time
import shutil
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

# Redirect database and FAISS paths to a test directory before importing the server app
TEST_DIR = Path(tempfile.mkdtemp())
settings.DB_PATH = TEST_DIR / "test_visionguard.db"
settings.VECTOR_INDEX_PATH = TEST_DIR / "test_index.faiss"
settings.REGISTRY_DIR = TEST_DIR / "registry"
settings.EVIDENCE_DIR = TEST_DIR / "evidence"

# Ensure directories exist
settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
settings.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

from api.server import app
from core.event_engine import EventEngine
from core.vector_db import VectorDBManager
from api.streams import LiveStreamWorker, ws_manager

client = TestClient(app)

def test_database_upgrades():
    """Verify that new schema tables exist (stream_health, recent_alerts, live_detections)."""
    engine = EventEngine(settings.DB_PATH)
    conn = sqlite3.connect(settings.DB_PATH)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    assert "stream_health" in tables
    assert "recent_alerts" in tables
    assert "live_detections" in tables
    conn.close()

def test_stream_health_diagnostics():
    """Verify that update_stream_health inserts and updates records correctly."""
    engine = EventEngine(settings.DB_PATH)
    engine.register_camera("CAM_HEALTH_TEST", "Health Room", "source.mp4")

    # Insert status
    success = engine.update_stream_health("CAM_HEALTH_TEST", 29.97, "ONLINE", 0)
    assert success is True

    # Retrieve and check
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM stream_health WHERE camera_id = ?;", ("CAM_HEALTH_TEST",)).fetchone()
    assert row is not None
    assert row["fps"] == 29.97
    assert row["status"] == "ONLINE"
    assert row["reconnect_count"] == 0
    conn.close()

def test_alert_cooldown_deduplication():
    """Verify that alert cooldown flags subsequent matches on the same camera as throttled."""
    engine = EventEngine(settings.DB_PATH)
    engine.register_person("P_COOL", "Cool Person", "Watchlist", "High")
    engine.register_camera("CAM_COOL", "Cool Corridor", "source.mp4")

    # First match: Cooldown is INACTIVE, returns False (fire alert)
    cooldown_1 = engine.check_alert_cooldown("P_COOL", "CAM_COOL", cooldown_seconds=5)
    assert cooldown_1 is False

    # Second match immediately: Cooldown is ACTIVE, returns True (throttle alert)
    cooldown_2 = engine.check_alert_cooldown("P_COOL", "CAM_COOL", cooldown_seconds=5)
    assert cooldown_2 is True

    # Sleep to expire cooldown
    time.sleep(6)

    # Third match after expiration: Cooldown is INACTIVE, returns False (fire alert)
    cooldown_3 = engine.check_alert_cooldown("P_COOL", "CAM_COOL", cooldown_seconds=5)
    assert cooldown_3 is False

def test_live_detections_history():
    """Verify that every match logs a historical trace in live_detections."""
    engine = EventEngine(settings.DB_PATH)
    engine.register_person("P_COOL", "Cool Person", "Watchlist", "High")
    engine.register_camera("CAM_COOL", "Cool Corridor", "source.mp4")

    success = engine.log_live_detection("DET_TEST_1", "P_COOL", "CAM_COOL", 0.85)
    assert success is True
    
    conn = sqlite3.connect(settings.DB_PATH)
    row = conn.execute("SELECT COUNT(*) FROM live_detections;").fetchone()
    assert row[0] == 1
    conn.close()

def test_multi_camera_escalation():
    """Verify that seen-across-multiple-cameras triggers escalation flag."""
    engine = EventEngine(settings.DB_PATH)
    engine.register_person("P_ESCALATE", "Escalating Target", "Watchlist", "High")
    engine.register_camera("CAM_A", "Cam A", "sourceA.mp4")
    engine.register_camera("CAM_B", "Cam B", "sourceB.mp4")

    # Initially, no prior sightings on other cameras
    escalated_1 = engine.check_multi_camera_sighting("P_ESCALATE", exclude_camera_id="CAM_A", window_minutes=5)
    assert escalated_1 is False

    # Log a sighting on Camera B
    engine.log_live_detection("DET_CAMB", "P_ESCALATE", "CAM_B", 0.90)

    # Query again from Camera A perspective. It should find the sighting on Camera B in the past 5 minutes
    escalated_2 = engine.check_multi_camera_sighting("P_ESCALATE", exclude_camera_id="CAM_A", window_minutes=5)
    assert escalated_2 is True

    # Query with a window that excludes it (e.g. past window is negative/zero)
    escalated_3 = engine.check_multi_camera_sighting("P_ESCALATE", exclude_camera_id="CAM_A", window_minutes=-1)
    assert escalated_3 is False

def test_websocket_connection():
    """Verify clients can connect to WebSocket channel and remain connected."""
    # TestClient ws connection manager
    with client.websocket_connect("/api/v1/surveillance/ws/alerts") as websocket:
        # Verify connection is successful and ws_manager registers it
        assert len(ws_manager.active_connections) == 1

def test_stream_worker_lifecycle():
    """Verify starting and stopping a live stream worker loop."""
    video_path = Path("sample_cctv.mp4")
    if not video_path.exists():
        # Skip video parsing part if sample is missing (checked in model downloads tests)
        return

    engine = EventEngine(settings.DB_PATH)
    vdb = VectorDBManager(settings.VECTOR_INDEX_PATH, 128, engine)

    # Load real detectors/recognizers
    from core.detector import FaceDetectorYuNet
    from core.recognizer import FaceRecognizerSFace
    
    detector = FaceDetectorYuNet(settings.YUNET_MODEL_PATH)
    recognizer = FaceRecognizerSFace(settings.SFACE_MODEL_PATH)

    worker = LiveStreamWorker(
        camera_id="CAM_LIVE_TEST",
        location="Front Gate Room",
        stream_source=str(video_path),
        threshold=0.40,
        detector=detector,
        recognizer=recognizer,
        event_engine=engine,
        vector_db=vdb,
        ws_broadcaster=ws_manager
    )

    # Start stream
    worker.start()
    time.sleep(2) # Let it run for 2 seconds to grab a few frames
    assert worker._running is True
    
    # Check diagnostics status was set to ONLINE
    conn = sqlite3.connect(settings.DB_PATH)
    status_row = conn.execute("SELECT status FROM stream_health WHERE camera_id = ?;", ("CAM_LIVE_TEST",)).fetchone()
    assert status_row is not None
    assert status_row[0] in ["ONLINE", "RECONNECTING"]
    conn.close()

    # Stop stream
    worker.stop()
    assert worker._running is False

def test_cleanup():
    try:
        shutil.rmtree(TEST_DIR)
    except Exception:
        pass
