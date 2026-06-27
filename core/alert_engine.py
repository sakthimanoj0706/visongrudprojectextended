import os
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

from config import settings
from core.event_engine import EventEngine

RISK_LEVEL_MAP = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4
}

class AlertCoordinator:
    def __init__(self, event_engine: EventEngine):
        self.event_engine = event_engine

    def calculate_severity_score(self, risk_level: str, similarity: float, escalated: bool) -> float:
        """
        Calculates a numeric severity score (0.0 to 1.0) based on target risk level,
        matching similarity, and multi-camera escalation status.
        """
        base_weight = {
            "Low": 0.2,
            "Medium": 0.5,
            "High": 0.8,
            "Critical": 1.0
        }.get(risk_level, 0.5)

        # Scale by similarity
        score = base_weight * similarity

        # Boost if escalated
        if escalated:
            score += 0.2

        return min(1.0, max(0.0, round(score, 2)))

    def evaluate_rules_and_dispatch(self, alert_id: str, person_id: str, name: str, risk_level: str,
                                   camera_id: str, location: str, similarity: float, escalated: bool,
                                   evidence_path: Optional[Path]) -> bool:
        """
        Checks active database alert rules against a match and dispatches notifications
        via configured channels (Webhooks, Mock Email, Mock SMS).
        """
        # Get active rules from DB
        rules = self.event_engine.get_alert_rules()
        
        # Filter active rules that match camera and threshold
        target_val = RISK_LEVEL_MAP.get(risk_level, 1)
        triggered_rules = []
        
        for rule in rules:
            if not rule.get("is_active", 1):
                continue
                
            # Rule camera check (NULL or exact match)
            rule_cam = rule.get("camera_id")
            if rule_cam and rule_cam != camera_id:
                continue
                
            # Rule risk threshold check
            threshold_val = RISK_LEVEL_MAP.get(rule.get("risk_level_threshold"), 1)
            if target_val >= threshold_val:
                triggered_rules.append(rule)

        if not triggered_rules:
            # If no custom rules are matched, we still create a default system alert for High/Critical risk targets
            if target_val >= RISK_LEVEL_MAP["High"]:
                # System fallback alert rule
                triggered_rules.append({
                    "rule_id": "SYS_FALLBACK",
                    "webhook_url": None
                })
            else:
                return False

        # If rules matched, calculate severity
        severity_score = self.calculate_severity_score(risk_level, similarity, escalated)
        
        # Link evidence paths: construct direct URL references
        crop_url = f"/evidence/{camera_id}/{alert_id}_crop.jpg"
        frame_url = f"/evidence/{camera_id}/{alert_id}_frame.jpg"
        annotated_url = f"/evidence/{camera_id}/{alert_id}_annotated.jpg"

        # Create alert entry in database (logs initial status ACTIVE)
        # Note: we store local target_dir in alerts evidence_path
        self.event_engine.create_alert(
            alert_id=alert_id,
            event_id=alert_id, # Alert ID maps 1-to-1 with Event ID for simplicity in real-time streams
            status="ACTIVE",
            severity_score=severity_score,
            evidence_path=str(evidence_path) if evidence_path else None
        )

        # Build notification payload
        payload = {
            "alert_id": alert_id,
            "person_id": person_id,
            "name": name,
            "risk_level": risk_level,
            "severity_score": severity_score,
            "camera_id": camera_id,
            "location": location,
            "timestamp": datetime.utcnow().isoformat(),
            "similarity": similarity,
            "escalated": escalated,
            "crop_url": crop_url,
            "frame_url": frame_url,
            "annotated_url": annotated_url
        }

        # Dispatch notifications
        for rule in triggered_rules:
            # 1. Outbound Webhook dispatch
            webhook_url = rule.get("webhook_url")
            if webhook_url:
                self._dispatch_webhook(webhook_url, payload)

        # 2. Mock Email dispatcher
        self._dispatch_mock_email(payload)

        # 3. Mock SMS dispatcher
        self._dispatch_mock_sms(payload)

        return True

    def _dispatch_webhook(self, url: str, payload: Dict[str, Any]):
        """Dispatches alert notification payload to external Webhook URL via HTTP POST."""
        try:
            print(f"[ALERT DISPATCH] Dispatching webhook to {url} ...")
            response = requests.post(url, json=payload, timeout=5)
            print(f"[ALERT DISPATCH] Webhook response code: {response.status_code}")
        except Exception as e:
            print(f"[ALERT DISPATCH ERROR] Webhook delivery failed to {url}: {e}")

    def _dispatch_mock_email(self, payload: Dict[str, Any]):
        """Mocks email notification dispatching by appending to a local mock file."""
        email_content = f"""
========================================
MOCK EMAIL NOTIFICATION
Date: {datetime.now().isoformat()}
To: {settings.MOCK_EMAIL_SINK}
Subject: [VisionGuard Alert] Watchlist Match - {payload['name']} ({payload['risk_level']})
----------------------------------------
Alert ID: {payload['alert_id']}
Camera: {payload['camera_id']} ({payload['location']})
Severity Score: {payload['severity_score']} (Escalated: {payload['escalated']})
Face Match Similarity: {payload['similarity']:.2f}
Evidence Images:
  - Face Crop: {payload['crop_url']}
  - Full Frame: {payload['frame_url']}
========================================
"""
        # Save to a mock text file inside outputs directory
        email_log = settings.OUTPUTS_DIR / "mock_email_notifications.log"
        try:
            with open(email_log, "a", encoding="utf-8") as f:
                f.write(email_content)
            print(f"[ALERT DISPATCH] Mock email successfully appended to {email_log}")
        except Exception as e:
            print(f"[ALERT DISPATCH ERROR] Failed to write mock email log: {e}")

    def _dispatch_mock_sms(self, payload: Dict[str, Any]):
        """Mocks SMS text alert dispatching to settings.MOCK_SMS_SINK."""
        sms_message = f"[VisionGuard Alert] {payload['name']} ({payload['risk_level']}) seen on Camera {payload['camera_id']}. Severity: {payload['severity_score']}. Crop: {payload['crop_url']}"
        sms_content = f"[{datetime.now().isoformat()}] SMS dispatched to {settings.MOCK_SMS_SINK}: {sms_message}\n"
        
        sms_log = settings.OUTPUTS_DIR / "mock_sms_notifications.log"
        try:
            with open(sms_log, "a", encoding="utf-8") as f:
                f.write(sms_content)
            print(f"[ALERT DISPATCH] Mock SMS successfully logged: {sms_message}")
        except Exception as e:
            print(f"[ALERT DISPATCH ERROR] Failed to write mock SMS log: {e}")
