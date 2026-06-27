import sys
import tempfile
import sqlite3
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.event_engine import EventEngine
from core.timeline import TimelineEngine

def test_database_and_timeline():
    # Create a temporary database path
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_visionguard.db"
        
        # 1. Initialize EventEngine
        engine = EventEngine(db_path)
        
        # Verify tables were created
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "cameras" in tables
        assert "persons" in tables
        assert "events" in tables
        assert "face_embeddings" in tables
        assert "search_audit_logs" in tables
        conn.close()

        # 2. Register Camera
        camera_registered = engine.register_camera(
            camera_id="CAM_TEST",
            location="Test Gate",
            video_source="test_source.mp4"
        )
        assert camera_registered is True

        # 3. Register Person
        person_registered = engine.register_person(
            person_id="P_TEST",
            name="Alice Tester",
            category="Test Watchlist",
            risk_level="High"
        )
        assert person_registered is True

        # Register face embedding mapping
        emb_registered = engine.register_face_embedding(
            embedding_id="EMB_TEST",
            person_id="P_TEST",
            vector_id=0,
            image_path="test_image.jpg",
            source_type="manual_enrollment",
            metadata={"notes": "Test vector"}
        )
        assert emb_registered is True

        # Retrieve person
        person = engine.get_person("P_TEST")
        assert person is not None
        assert person["name"] == "Alice Tester"
        assert person["risk_level"] == "High"
        assert len(person["embeddings"]) == 1
        assert person["embeddings"][0]["vector_id"] == 0

        # 4. Log Event
        match_details = {"match": True, "confidence": 0.85, "threshold": 0.60}
        event_logged = engine.log_event(
            event_id="EVT_TEST_1",
            person_id="P_TEST",
            camera_id="CAM_TEST",
            video_source="test_source.mp4",
            timestamp="00:01:10",
            frame_number=2100,
            confidence=0.85,
            bounding_box=[10, 20, 100, 100],
            match_details=match_details
        )
        assert event_logged is True

        # Log second event
        event_logged_2 = engine.log_event(
            event_id="EVT_TEST_2",
            person_id="P_TEST",
            camera_id="CAM_TEST",
            video_source="test_source.mp4",
            timestamp="00:02:15",
            frame_number=4050,
            confidence=0.92,
            bounding_box=[15, 25, 110, 110],
            match_details={"match": True, "confidence": 0.92, "threshold": 0.60}
        )
        assert event_logged_2 is True

        # Retrieve events
        events = engine.get_events_for_person("P_TEST")
        assert len(events) == 2
        assert events[0]["event_id"] == "EVT_TEST_1"
        assert events[0]["bounding_box"] == [10, 20, 100, 100]
        assert events[1]["event_id"] == "EVT_TEST_2"

        # 5. Timeline Generation
        timeline_engine = TimelineEngine(engine)
        timeline = timeline_engine.generate_timeline("P_TEST")
        
        assert timeline["person_id"] == "P_TEST"
        assert timeline["person_name"] == "Alice Tester"
        assert timeline["total_sightings"] == 2
        assert len(timeline["sightings"]) == 2
        assert timeline["sightings"][0]["timestamp"] == "00:01:10"
        assert timeline["sightings"][1]["timestamp"] == "00:02:15"
        
        # Test reports output structure
        json_report = Path(tmpdir) / "timeline.json"
        md_report = Path(tmpdir) / "timeline.md"
        timeline_engine.save_timeline_report(timeline, json_report, md_report)
        
        assert json_report.exists()
        assert md_report.exists()
        
        with open(json_report, "r") as f:
            data = json.load(f)
            assert data["person_id"] == "P_TEST"
            assert data["total_sightings"] == 2
            
        with open(md_report, "r") as f:
            content = f.read()
            assert "Alice Tester" in content
            assert "Test Watchlist" in content
            assert "CAM_TEST" in content
            assert "00:01:10" in content
            assert "00:02:15" in content
