from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str

class HealthStatus(BaseModel):
    status: str
    watchlist_size: int
    events_logged: int
    faiss_vectors: int
    timestamp: str

class AlertCreate(BaseModel):
    person_id: str
    camera_id: str
    risk_level: str = Field(..., pattern="^(Low|Medium|High|Critical)$")
    timestamp: Optional[str] = None
    notes: Optional[str] = None

class AlertResponse(BaseModel):
    alert_id: str
    person_id: str
    camera_id: str
    risk_level: str
    timestamp: str
    notes: Optional[str]

class JobStatus(BaseModel):
    job_id: str
    status: str # queued, running, completed, failed
    progress: float # 0.0 to 1.0
    video_source: str
    matches_found: int
    error: Optional[str] = None
    created_at: str

class SearchSighting(BaseModel):
    event_id: str
    camera_id: str
    camera_location: str
    video_source: str
    timestamp: str
    frame_number: int
    confidence: float
    bounding_box: List[int]
    match_details: Dict[str, Any]

class SightingResponse(BaseModel):
    event_id: str
    person_id: str
    camera_id: str
    video_source: str
    timestamp: str
    frame_number: int
    confidence: float
    bounding_box: List[int]
    match_details: Dict[str, Any]

class TimelineResponse(BaseModel):
    person_id: str
    person_name: str
    category: str
    total_sightings: int
    cameras_visited_count: int
    cameras_visited: List[str]
    sightings: List[SearchSighting]


# --- Phase 5 Schemas: Alert Rules and Alert Workflow Operations ---

class AlertRuleCreate(BaseModel):
    camera_id: Optional[str] = None
    risk_level_threshold: str = Field(..., pattern="^(Low|Medium|High|Critical)$")
    webhook_url: Optional[str] = None
    is_active: Optional[bool] = True

class AlertRuleResponse(BaseModel):
    rule_id: str
    camera_id: Optional[str]
    risk_level_threshold: str
    webhook_url: Optional[str]
    is_active: bool

class AlertStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(ACKNOWLEDGED|RESOLVED|FALSE_POSITIVE)$")
    notes: Optional[str] = None

class AlertLifecycleTimelineEntry(BaseModel):
    status: str
    operator: Optional[str]
    notes: Optional[str]
    timestamp: str

class AlertDetailsResponse(BaseModel):
    alert_id: str
    event_id: str
    status: str
    severity_score: float
    evidence_path: Optional[str]
    assigned_operator: Optional[str]
    resolution_notes: Optional[str]
    created_at: str
    updated_at: str
    person_id: str
    camera_id: str
    sighting_time: str
    person_name: str

class FalsePositiveAnalyticsResponse(BaseModel):
    total_alerts: int
    active_count: int
    acknowledged_count: int
    resolved_count: int
    false_positive_count: int
    false_positive_rate: float
    false_positives_by_camera: List[Dict[str, Any]]
    false_positives_by_person: List[Dict[str, Any]]

class MemoryQueryRequest(BaseModel):
    query_text: str
    k: Optional[int] = 5
    person_id: Optional[str] = None
    camera_id: Optional[str] = None

class MemoryResponse(BaseModel):
    memory_id: str
    reference_id: str
    entity_type: str
    person_id: Optional[str]
    camera_id: Optional[str]
    timestamp: str
    document_text: str
    evidence_path: Optional[str]
    similarity: float

class ChatHistoryEntry(BaseModel):
    role: str # "user" or "assistant" / "model"
    content: str

class AssistantChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatHistoryEntry]] = None

class AssistantChatResponse(BaseModel):
    response: str
    sources: List[MemoryResponse]
    backend_used: str


