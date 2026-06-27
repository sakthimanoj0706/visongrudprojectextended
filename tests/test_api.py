import sys
import tempfile
import sqlite3
import shutil
import numpy as np
import cv2
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

# Temporarily redirect database and FAISS paths to a test directory before importing the server app
# to prevent tests from altering development databases
TEST_DIR = Path(tempfile.mkdtemp())
settings.DB_PATH = TEST_DIR / "test_visionguard.db"
settings.VECTOR_INDEX_PATH = TEST_DIR / "test_index.faiss"
settings.REGISTRY_DIR = TEST_DIR / "registry"

# Ensure directories exist
settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)

from api.server import app

client = TestClient(app)

# Helper to get JWT token
def get_auth_token(username: str, password: str) -> str:
    response = client.post(
        "/api/v1/auth/token",
        json={"username": username, "password": password}
    )
    assert response.status_code == 200
    return response.json()["access_token"]

def create_mock_face_image(path: Path):
    """Creates a blank image or copies a real one if available."""
    real_target = Path("target_face.jpg")
    if real_target.exists():
        shutil.copy(real_target, path)
    else:
        # Create a mock black image of 300x300 (fails face detection but tests image upload parsing)
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.imwrite(str(path), img)

def test_api_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "watchlist_size" in data
    assert "faiss_vectors" in data

def test_api_auth():
    # Valid login
    response = client.post(
        "/api/v1/auth/token",
        json={"username": "admin_user", "password": "admin_pass"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["role"] == "Admin"

    # Invalid login
    response = client.post(
        "/api/v1/auth/token",
        json={"username": "admin_user", "password": "wrong_pass"}
    )
    assert response.status_code == 401

def test_api_rbac_permissions():
    # Get tokens
    admin_token = get_auth_token("admin_user", "admin_pass")
    auditor_token = get_auth_token("auditor_user", "auditor_pass")

    # Auditor should not have access to Watchlist Enrollment
    test_img_path = TEST_DIR / "temp_face.jpg"
    create_mock_face_image(test_img_path)
    
    with open(test_img_path, "rb") as img_file:
        response = client.post(
            "/api/v1/watchlist/enroll",
            headers={"Authorization": f"Bearer {auditor_token}"},
            data={"name": "Alice Auditor", "category": "Test", "risk_level": "High"},
            files={"image": ("temp_face.jpg", img_file, "image/jpeg")}
        )
    # Expected 403 Forbidden
    assert response.status_code == 403

    # Auditor should have access to Audit Logs
    response = client.get(
        "/api/v1/watchlist/audit-logs",
        headers={"Authorization": f"Bearer {auditor_token}"}
    )
    assert response.status_code == 200

def test_api_enrollment_and_search():
    admin_token = get_auth_token("admin_user", "admin_pass")
    
    # We must copy target_face.jpg to ensure YuNet finds a face and registers it
    target_img = Path("target_face.jpg")
    if not target_img.exists():
        # Skip this test if target_face.jpg was not created during manual verification
        print("[WARN] target_face.jpg missing. Skipping E2E API tests.")
        return

    # 1. Test Watchlist Enrollment (Admin)
    # Auto-generation test: leave person_id blank
    with open(target_img, "rb") as img_file:
        response = client.post(
            "/api/v1/watchlist/enroll",
            headers={"Authorization": f"Bearer {admin_token}"},
            data={"name": "Jane Watchlist", "category": "Watchlist", "risk_level": "Critical"},
            files={"image": ("target_face.jpg", img_file, "image/jpeg")}
        )
    assert response.status_code == 200
    enroll_data = response.json()
    assert enroll_data["status"] == "success"
    assert enroll_data["name"] == "Jane Watchlist"
    assert enroll_data["risk_level"] == "Critical"
    person_id = enroll_data["person_id"]
    assert person_id.startswith("P_") # Auto-generated ID

    # 2. Test Image-based Watchlist Search
    with open(target_img, "rb") as img_file:
        response = client.post(
            "/api/v1/watchlist/search/image",
            headers={"Authorization": f"Bearer {admin_token}"},
            data={"threshold": 0.40, "k": 2},
            files={"image": ("target_face.jpg", img_file, "image/jpeg")}
        )
    assert response.status_code == 200
    search_data = response.json()
    assert search_data["status"] == "success"
    assert search_data["detected_faces_count"] > 0
    assert len(search_data["matches"]) > 0
    assert search_data["matches"][0]["person_id"] == person_id
    assert search_data["matches"][0]["risk_level"] == "Critical"

    # 3. Test Sighting Timeline Sighting Query
    response = client.get(
        f"/api/v1/watchlist/timeline/{person_id}",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    timeline = response.json()
    assert timeline["person_id"] == person_id
    assert timeline["person_name"] == "Jane Watchlist"

    # 4. Test RAG Event Sighting Query
    response = client.get(
        f"/api/v1/persons/{person_id}/events",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_api_alert_trigger():
    admin_token = get_auth_token("admin_user", "admin_pass")
    
    # We must register a person first in the TEST DB to refer to it
    response = client.post(
        "/api/v1/watchlist/enroll",
        headers={"Authorization": f"Bearer {admin_token}"},
        data={"name": "Alert Person", "category": "Watchlist", "risk_level": "High", "person_id": "P_ALERT_1"},
        files={"image": ("target_face.jpg", open("target_face.jpg", "rb"), "image/jpeg")}
    )
    assert response.status_code == 200

    response = client.post(
        "/api/v1/alerts",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "person_id": "P_ALERT_1",
            "camera_id": "CAM_ALERT_1",
            "risk_level": "Critical",
            "notes": "Manual alert trigger test"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["alert_id"].startswith("ALT_")
    assert data["person_id"] == "P_ALERT_1"
    assert data["risk_level"] == "Critical"

def test_api_background_job():
    admin_token = get_auth_token("admin_user", "admin_pass")
    
    # Trigger video search job
    video_path = Path("sample_cctv.mp4")
    if not video_path.exists():
        return # Skip if video is not available

    response = client.post(
        "/api/v1/surveillance/search-video",
        headers={"Authorization": f"Bearer {admin_token}"},
        params={
            "video_path": str(video_path),
            "camera_id": "CAM_JOB_1",
            "camera_location": "Job Corridor",
            "target_id": "P_ALERT_1",
            "threshold": 0.40,
            "frame_skip": 10,
            "resize_width": 640
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    job_id = data["job_id"]
    assert job_id.startswith("JOB_")

    # Get job status immediately
    response = client.get(
        f"/api/v1/surveillance/jobs/{job_id}",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    job_data = response.json()
    assert job_data["job_id"] == job_id
    assert job_data["status"] in ["queued", "running", "completed"]

# Cleanup temp files
def test_cleanup():
    try:
        shutil.rmtree(TEST_DIR)
    except Exception:
        pass
