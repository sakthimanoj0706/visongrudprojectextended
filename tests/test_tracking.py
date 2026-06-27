import sys
import tempfile
import sqlite3
import json
import time
import numpy as np
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings
from core.event_engine import EventEngine
from core.tracker import SingleCameraTracker, Tracklet
from core.reid import PersonReIDExtractor, YoloPersonDetector, estimate_body_box

def test_body_crop_estimation():
    # Test case 1: Standard face within boundaries
    face_box = [100, 100, 50, 50]
    frame_shape = (720, 1280, 3)
    
    body_box = estimate_body_box(face_box, frame_shape)
    
    # Expected estimation based on:
    # bx = fx - fw * 1.5 = 100 - 75 = 25
    # by = fy - fh * 0.2 = 100 - 10 = 90
    # bw = fw * 4.0 = 200
    # bh = fh * 7.5 = 375
    assert body_box == [25, 90, 200, 375]
    
    # Test case 2: Near boundary clamping
    face_box = [10, 10, 20, 20]
    body_box = estimate_body_box(face_box, frame_shape)
    assert body_box[0] >= 0
    assert body_box[1] >= 0
    assert body_box[2] <= 1280
    assert body_box[3] <= 720

def test_single_camera_tracker():
    tracker = SingleCameraTracker(camera_id="CAM_1", timeout_seconds=1.0)
    
    # Frame 1: Initial detection
    det_1 = [{"box": [100, 100, 50, 50], "person_id": None}]
    active_matches, expired = tracker.update(det_1)
    
    assert len(active_matches) == 1
    assert len(expired) == 0
    tracklet_1, _ = active_matches[0]
    tid_1 = tracklet_1.tracklet_id
    assert tracklet_1.camera_id == "CAM_1"
    assert tracklet_1.person_id is None
    
    # Frame 2: Move slightly (should associate via IoU)
    det_2 = [{"box": [105, 102, 52, 48], "person_id": "P001"}]
    active_matches, expired = tracker.update(det_2)
    assert len(active_matches) == 1
    tracklet_2, match_det = active_matches[0]
    assert tracklet_2.tracklet_id == tid_1
    # Check that matched detection carries identity
    assert match_det["person_id"] == "P001"
    
    # Frame 3: Move to completely different location (should spawn new tracklet)
    det_3 = [{"box": [500, 500, 50, 50], "person_id": None}]
    active_matches, expired = tracker.update(det_3)
    assert len(active_matches) == 1
    tracklet_3, _ = active_matches[0]
    assert tracklet_3.tracklet_id != tid_1
    
    # Frame 4: Test timeout/expiration
    time.sleep(1.1)
    active_matches, expired = tracker.update([])
    assert len(active_matches) == 0
    assert len(expired) == 2  # Both old tracklets should have expired

def test_reid_extractor_features():
    # If the model does not exist, skip the test gracefully
    if not settings.REID_MODEL_PATH.exists():
        pytest.skip("Tencent Youtu ReID model weights missing.")

    extractor = PersonReIDExtractor(settings.REID_MODEL_PATH)
    
    # Create dummy crop image (128x256x3)
    dummy_crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    
    vector = extractor.extract_embedding(dummy_crop)
    assert vector.shape == (768,)
    
    # Verify L2 normalization
    norm = np.linalg.norm(vector)
    assert pytest.approx(norm, 1e-5) == 1.0
    
    # Compare similarity
    dummy_crop_2 = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    vector_2 = extractor.extract_embedding(dummy_crop_2)
    
    sim = extractor.compute_similarity(vector, vector_2)
    assert -1.0 <= sim <= 1.0

def test_tracking_database_ops():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_vg_tracking.db"
        engine = EventEngine(db_path)
        
        # Register entities
        assert engine.register_camera("CAM_A", "Entrance Gate", "source_a.mp4") is True
        assert engine.register_camera("CAM_B", "Main Lobby", "source_b.mp4") is True
        assert engine.register_person("P001", "John Doe", "Watchlist", "High") is True
        
        # 1. Create tracklets
        tid = "TRK_TEST_123"
        start_time = "2026-06-27T10:00:00Z"
        assert engine.create_tracklet(tid, "CAM_A", start_time, None) is True
        
        # 2. Update tracklet person ID
        assert engine.update_tracklet_person(tid, "P001") is True
        
        # 3. Register ReID features
        vector = [0.1] * 768
        reid_id = "REID_123"
        assert engine.register_reid_embedding(reid_id, tid, vector) is True
        
        # Verify fetching recent ReID embeddings
        recent = engine.get_recent_reid_embeddings(limit_minutes=10)
        assert len(recent) == 1
        assert recent[0]["tracklet_id"] == tid
        assert recent[0]["camera_id"] == "CAM_A"
        assert recent[0]["person_id"] == "P001"
        assert len(recent[0]["vector"]) == 768
        assert pytest.approx(recent[0]["vector"][0], 1e-5) == 0.1
        
        # 4. Close tracklet
        assert engine.close_tracklet(tid, "2026-06-27T10:00:05Z") is True
        
        # 5. Log movements (Transitions)
        assert engine.register_movement(
            person_id="P001",
            from_camera_id="CAM_A",
            to_camera_id="CAM_B",
            departure_time="2026-06-27T10:00:05Z",
            arrival_time="2026-06-27T10:00:15Z",
            duration_seconds=10.0,
            similarity=0.82
        ) is not None

        
        # Fetch movements
        movements = engine.get_movements("P001")
        assert len(movements) == 1
        assert movements[0]["from_camera_id"] == "CAM_A"
        assert movements[0]["to_camera_id"] == "CAM_B"
        assert movements[0]["duration_seconds"] == 10.0
        assert movements[0]["similarity"] == 0.82
        
        # 6. Fetch camera visit count (heatmap prep)
        visits = engine.get_camera_visit_counts()
        assert len(visits) >= 2
        # Verify keys
        assert "camera_id" in visits[0]
        assert "location" in visits[0]
        assert "visit_count" in visits[0]
