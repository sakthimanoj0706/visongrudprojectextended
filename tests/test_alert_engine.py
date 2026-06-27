import sys
import tempfile
import sqlite3
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

# Redirect database and index paths to a test directory before importing engines
TEST_DIR = Path(tempfile.mkdtemp())
settings.DB_PATH = TEST_DIR / "test_visionguard.db"
settings.VECTOR_INDEX_PATH = TEST_DIR / "test_index.faiss"
settings.REGISTRY_DIR = TEST_DIR / "registry"
settings.EVIDENCE_DIR = TEST_DIR / "evidence"
settings.OUTPUTS_DIR = TEST_DIR / "outputs"

# Ensure directories exist
settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
settings.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

from core.event_engine import EventEngine
from core.alert_engine import AlertCoordinator

def test_database_alert_schema():
    """Verify that alert engine schema tables exist (alert_rules, alerts, alert_lifecycle)."""
    engine = EventEngine(settings.DB_PATH)
    conn = sqlite3.connect(settings.DB_PATH)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    assert "alert_rules" in tables
    assert "alerts" in tables
    assert "alert_lifecycle" in tables
    conn.close()

def test_alert_rule_crud():
    """Verify alert rule creation, retrieval, status checking, and deletion."""
    engine = EventEngine(settings.DB_PATH)
    
    # 1. Create Rule
    success = engine.create_alert_rule(
        rule_id="RULE_TEST_01",
        camera_id="CAM_RULE_TEST",
        risk_level_threshold="High",
        webhook_url="https://webhook.site/test",
        is_active=1
    )
    assert success is True
    
    # 2. Get Rules
    rules = engine.get_alert_rules()
    assert len(rules) >= 1
    rule_dict = next(r for r in rules if r["rule_id"] == "RULE_TEST_01")
    assert rule_dict["camera_id"] == "CAM_RULE_TEST"
    assert rule_dict["risk_level_threshold"] == "High"
    assert rule_dict["webhook_url"] == "https://webhook.site/test"
    assert rule_dict["is_active"] == 1
    
    # 3. Get Single Rule
    rule = engine.get_alert_rule("RULE_TEST_01")
    assert rule is not None
    assert rule["camera_id"] == "CAM_RULE_TEST"
    
    # 4. Delete Rule
    del_success = engine.delete_alert_rule("RULE_TEST_01")
    assert del_success is True
    assert engine.get_alert_rule("RULE_TEST_01") is None

def test_alert_creation_and_transitions():
    """Verify alert logs, status transitions, and lifecycle timeline updates."""
    engine = EventEngine(settings.DB_PATH)
    
    # Setup camera, person, and event to satisfy foreign key constraints
    engine.register_camera("CAM_TRANS_TEST", "Transition Corridor", "source.mp4")
    engine.register_person("P_TRANS", "Transition Person", "Watchlist", "Medium")
    
    # Log Sighting Event
    event_id = "EVT_TRANS_TEST"
    engine.log_event(
        event_id=event_id,
        person_id="P_TRANS",
        camera_id="CAM_TRANS_TEST",
        video_source="LIVE",
        timestamp=datetime.utcnow().isoformat(),
        frame_number=-1,
        confidence=0.85,
        bounding_box=[10, 20, 30, 40],
        match_details={"risk_level": "Medium"}
    )
    
    # 1. Create Alert
    alert_id = "ALT_TRANS_TEST"
    evidence_path = settings.EVIDENCE_DIR / "CAM_TRANS_TEST/test_evidence"
    success = engine.create_alert(
        alert_id=alert_id,
        event_id=event_id,
        status="ACTIVE",
        severity_score=0.68,
        evidence_path=str(evidence_path)
    )
    assert success is True
    
    # Verify Alert details
    alert = engine.get_alert_with_event(alert_id)
    assert alert is not None
    assert alert["status"] == "ACTIVE"
    assert alert["severity_score"] == 0.68
    assert alert["evidence_path"] == str(evidence_path)
    
    # 2. Transition Alert status: ACTIVE -> ACKNOWLEDGED
    transition_1 = engine.transition_alert_status(
        alert_id=alert_id,
        new_status="ACKNOWLEDGED",
        operator="operator_bob",
        notes="Operator acknowledging match."
    )
    assert transition_1 is True
    
    alert = engine.get_alert_with_event(alert_id)
    assert alert["status"] == "ACKNOWLEDGED"
    assert alert["assigned_operator"] == "operator_bob"
    
    # 3. Transition Alert status: ACKNOWLEDGED -> RESOLVED
    transition_2 = engine.transition_alert_status(
        alert_id=alert_id,
        new_status="RESOLVED",
        operator="operator_bob",
        notes="Person identified and resolved."
    )
    assert transition_2 is True
    
    alert = engine.get_alert_with_event(alert_id)
    assert alert["status"] == "RESOLVED"
    assert alert["resolution_notes"] == "Person identified and resolved."
    
    # 4. Check Alert Lifecycle timeline
    timeline = engine.get_alert_lifecycle_timeline(alert_id)
    assert len(timeline) == 3 # ACTIVE init, ACKNOWLEDGED, RESOLVED
    assert timeline[0]["status"] == "ACTIVE"
    assert timeline[1]["status"] == "ACKNOWLEDGED"
    assert timeline[1]["operator"] == "operator_bob"
    assert timeline[2]["status"] == "RESOLVED"

def test_alert_coordinator_severity_and_rules():
    """Verify rule threshold matching logic and severity scoring formulas."""
    engine = EventEngine(settings.DB_PATH)
    coordinator = AlertCoordinator(engine)
    
    # Setup test cameras/persons
    engine.register_camera("CAM_EVAL_A", "Entrance Gate A", "gateA.mp4")
    engine.register_camera("CAM_EVAL_B", "Entrance Gate B", "gateB.mp4")
    engine.register_person("P_EVAL_1", "Low Risk Target", "Watchlist", "Low")
    engine.register_person("P_EVAL_2", "Critical Risk Target", "Watchlist", "Critical")
    
    # Log Events
    engine.log_event("ALT_EVAL_1", "P_EVAL_1", "CAM_EVAL_A", "LIVE", "00:00:01", -1, 0.90, [0,0,0,0], {"risk_level": "Low"})
    engine.log_event("ALT_EVAL_2", "P_EVAL_2", "CAM_EVAL_B", "LIVE", "00:00:01", -1, 0.90, [0,0,0,0], {"risk_level": "Critical"})
    
    # 1. Test Severity Scoring formula
    score_low = coordinator.calculate_severity_score(risk_level="Low", similarity=0.80, escalated=False)
    # Low weight: 0.2. Score = 0.2 * 0.8 = 0.16
    assert score_low == 0.16
    
    score_critical_esc = coordinator.calculate_severity_score(risk_level="Critical", similarity=0.90, escalated=True)
    # Critical weight: 1.0. Score = (1.0 * 0.9) + 0.2 = 1.1 -> capped at 1.0
    assert score_critical_esc == 1.0
    
    # 2. Test Rules Evaluation Matching
    # Clear rules
    for r in engine.get_alert_rules():
        engine.delete_alert_rule(r["rule_id"])
        
    # Add a High threshold rule for CAM_EVAL_B
    engine.create_alert_rule("RULE_CRIT", "CAM_EVAL_B", "High", "https://webhook.site/crit", 1)
    
    # Evaluate P_EVAL_1 (Low risk) on CAM_EVAL_A. It should fail to trigger (Low < High threshold, and no system fallback for Low)
    triggered_1 = coordinator.evaluate_rules_and_dispatch(
        alert_id="ALT_EVAL_1",
        person_id="P_EVAL_1",
        name="Low Target",
        risk_level="Low",
        camera_id="CAM_EVAL_A",
        location="Gate A",
        similarity=0.90,
        escalated=False,
        evidence_path=None
    )
    assert triggered_1 is False
    
    # Evaluate P_EVAL_2 (Critical risk) on CAM_EVAL_B. It should match rule RULE_CRIT (Critical >= High)
    triggered_2 = coordinator.evaluate_rules_and_dispatch(
        alert_id="ALT_EVAL_2",
        person_id="P_EVAL_2",
        name="Critical Target",
        risk_level="Critical",
        camera_id="CAM_EVAL_B",
        location="Gate B",
        similarity=0.95,
        escalated=False,
        evidence_path=None
    )
    assert triggered_2 is True
    
    # Check that alert was registered in database
    alert = engine.get_alert_with_event("ALT_EVAL_2")
    assert alert is not None
    assert alert["status"] == "ACTIVE"

def test_mock_dispatches():
    """Verify that email and SMS mock dispatches write to notification logs."""
    engine = EventEngine(settings.DB_PATH)
    coordinator = AlertCoordinator(engine)
    
    payload = {
        "alert_id": "ALT_MOCK_DISPATCH",
        "person_id": "P_MOCK",
        "name": "Jane Dispatcher",
        "risk_level": "High",
        "severity_score": 0.85,
        "camera_id": "CAM_MOCK",
        "location": "Mock Room",
        "timestamp": datetime.utcnow().isoformat(),
        "similarity": 0.91,
        "escalated": False,
        "crop_url": "/mock/crop.jpg",
        "frame_url": "/mock/frame.jpg",
        "annotated_url": "/mock/annotated.jpg"
    }
    
    # Trigger mock email
    coordinator._dispatch_mock_email(payload)
    email_log = settings.OUTPUTS_DIR / "mock_email_notifications.log"
    assert email_log.exists()
    assert "To: alerts@visionguard.local" in email_log.read_text(encoding="utf-8")
    
    # Trigger mock SMS
    coordinator._dispatch_mock_sms(payload)
    sms_log = settings.OUTPUTS_DIR / "mock_sms_notifications.log"
    assert sms_log.exists()
    assert "dispatched to +15550199" in sms_log.read_text(encoding="utf-8")

def test_false_positive_analytics():
    """Verify calculations of false positive rates and aggregates by camera/person."""
    engine = EventEngine(settings.DB_PATH)
    
    # Clear alerts
    conn = sqlite3.connect(settings.DB_PATH)
    conn.execute("DELETE FROM alerts;")
    conn.execute("DELETE FROM alert_lifecycle;")
    conn.commit()
    conn.close()
    
    # Create test events
    engine.register_camera("CAM_FP_1", "FP Lobby", "lobby.mp4")
    engine.register_person("P_FP_1", "Target One", "Watchlist", "High")
    
    # Add multiple alerts
    # Alert 1: RESOLVED
    engine.log_event("EVT_FP_1", "P_FP_1", "CAM_FP_1", "LIVE", "00:00:01", -1, 0.90, [0,0,0,0], {"risk": "High"})
    engine.create_alert("ALT_FP_1", "EVT_FP_1", "RESOLVED", 0.85, None)
    
    # Alert 2: FALSE_POSITIVE
    engine.log_event("EVT_FP_2", "P_FP_1", "CAM_FP_1", "LIVE", "00:00:01", -1, 0.90, [0,0,0,0], {"risk": "High"})
    engine.create_alert("ALT_FP_2", "EVT_FP_2", "FALSE_POSITIVE", 0.70, None)
    # Re-insert transition to update status
    engine.transition_alert_status("ALT_FP_2", "FALSE_POSITIVE", "op_bob", "Not matching.")
    
    # Alert 3: RESOLVED
    engine.log_event("EVT_FP_3", "P_FP_1", "CAM_FP_1", "LIVE", "00:00:01", -1, 0.90, [0,0,0,0], {"risk": "High"})
    engine.create_alert("ALT_FP_3", "EVT_FP_3", "RESOLVED", 0.90, None)
    
    # Alert 4: ACTIVE
    engine.log_event("EVT_FP_4", "P_FP_1", "CAM_FP_1", "LIVE", "00:00:01", -1, 0.90, [0,0,0,0], {"risk": "High"})
    engine.create_alert("ALT_FP_4", "EVT_FP_4", "ACTIVE", 0.75, None)
    
    # Query false positive analytics
    analytics = engine.get_false_positive_analytics()
    assert analytics["total_alerts"] == 4
    assert analytics["resolved_count"] == 2
    assert analytics["false_positive_count"] == 1
    # 1 FP / (2 RESOLVED + 1 FP) = 0.3333
    assert analytics["false_positive_rate"] == 0.3333
    
    # Verify camera breakdown
    cam_breakdown = analytics["false_positives_by_camera"]
    assert len(cam_breakdown) == 1
    assert cam_breakdown[0]["camera_id"] == "CAM_FP_1"
    assert cam_breakdown[0]["count"] == 1
    
    # Verify person breakdown
    person_breakdown = analytics["false_positives_by_person"]
    assert len(person_breakdown) == 1
    assert person_breakdown[0]["person_id"] == "P_FP_1"
    assert person_breakdown[0]["count"] == 1

def test_alert_retention_archiving():
    """Verify that resolved alerts older than retention are deleted while active alerts are preserved."""
    engine = EventEngine(settings.DB_PATH)
    
    # Reset database alerts
    conn = sqlite3.connect(settings.DB_PATH)
    conn.execute("DELETE FROM alerts;")
    conn.execute("DELETE FROM alert_lifecycle;")
    conn.commit()
    conn.close()
    
    # Setup test event
    engine.register_camera("CAM_RET", "Archive Gate", "gate.mp4")
    engine.register_person("P_RET", "Archive Person", "Watchlist", "High")
    engine.log_event("EVT_RET_1", "P_RET", "CAM_RET", "LIVE", "00:00:01", -1, 0.90, [0,0,0,0], {"risk": "High"})
    engine.log_event("EVT_RET_2", "P_RET", "CAM_RET", "LIVE", "00:00:01", -1, 0.90, [0,0,0,0], {"risk": "High"})
    engine.log_event("EVT_RET_3", "P_RET", "CAM_RET", "LIVE", "00:00:01", -1, 0.90, [0,0,0,0], {"risk": "High"})
    
    # 1. Old Resolved Alert (Created 40 days ago) -> Should be deleted
    engine.create_alert("ALT_OLD_RESOLVED", "EVT_RET_1", "RESOLVED", 0.80, None)
    
    # 2. Old Active Alert (Created 40 days ago) -> Should be preserved
    engine.create_alert("ALT_OLD_ACTIVE", "EVT_RET_2", "ACTIVE", 0.85, None)
    
    # 3. New Resolved Alert (Created 2 days ago) -> Should be preserved
    engine.create_alert("ALT_NEW_RESOLVED", "EVT_RET_3", "RESOLVED", 0.90, None)
    
    # Manually adjust created_at timestamps in database to mock old alerts
    old_time_str = (datetime.utcnow() - timedelta(days=40)).isoformat()
    new_time_str = (datetime.utcnow() - timedelta(days=2)).isoformat()
    
    conn = sqlite3.connect(settings.DB_PATH)
    conn.execute("UPDATE alerts SET created_at = ? WHERE alert_id = ?;", (old_time_str, "ALT_OLD_RESOLVED"))
    conn.execute("UPDATE alerts SET created_at = ? WHERE alert_id = ?;", (old_time_str, "ALT_OLD_ACTIVE"))
    conn.execute("UPDATE alerts SET created_at = ? WHERE alert_id = ?;", (new_time_str, "ALT_NEW_RESOLVED"))
    conn.commit()
    conn.close()
    
    # Perform archive cleanup for alerts older than 30 days
    archived = engine.archive_old_alerts(retention_days=30)
    assert archived == 1 # Only ALT_OLD_RESOLVED should be deleted
    
    # Check what remains
    assert engine.get_alert_with_event("ALT_OLD_RESOLVED") is None
    assert engine.get_alert_with_event("ALT_OLD_ACTIVE") is not None
    assert engine.get_alert_with_event("ALT_NEW_RESOLVED") is not None

def test_cleanup():
    try:
        shutil.rmtree(TEST_DIR)
    except Exception:
        pass
