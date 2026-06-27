import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

class EventEngine:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection to the SQLite database. Enables foreign key support."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes the database schema with all tables, constraints, and indexes."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check and migrate cameras table if it has unique/UNIQUE constraint on video_source
        if self.db_path.exists():
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='cameras';")
                    row = cursor.fetchone()
                    if row and "unique" in row[0].lower():
                        print("[DATABASE] Migrating 'cameras' table to remove UNIQUE constraint...")
                        conn.execute("PRAGMA foreign_keys = OFF;")
                        conn.execute("ALTER TABLE cameras RENAME TO _cameras_old;")
                        conn.execute("""
                            CREATE TABLE cameras (
                                camera_id TEXT PRIMARY KEY,
                                location TEXT NOT NULL,
                                video_source TEXT NOT NULL
                            );
                        """)
                        conn.execute("INSERT OR IGNORE INTO cameras (camera_id, location, video_source) SELECT camera_id, location, video_source FROM _cameras_old;")
                        conn.execute("DROP TABLE _cameras_old;")
                        conn.commit()
                        print("[DATABASE] Migration completed successfully.")
            except Exception as e:
                print(f"[DATABASE WARNING] Migration check failed: {e}")

        with self._get_connection() as conn:
            # Table: cameras
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cameras (
                    camera_id TEXT PRIMARY KEY,
                    location TEXT NOT NULL,
                    video_source TEXT NOT NULL
                );
            """)

            # Table: persons
            conn.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    person_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)

            # Table: face_embeddings
            conn.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    embedding_id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL,
                    vector_id INTEGER NOT NULL UNIQUE,
                    image_path TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE
                );
            """)

            # Table: search_audit_logs
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_audit_logs (
                    log_id TEXT PRIMARY KEY,
                    query_timestamp TEXT NOT NULL,
                    query_source TEXT NOT NULL,
                    query_params TEXT NOT NULL,
                    results_count INTEGER NOT NULL,
                    execution_time_ms REAL NOT NULL
                );
            """)

            # Table: events (Sighting alerts logs)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    video_source TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    frame_number INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    bounding_box TEXT NOT NULL,
                    match_details TEXT NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES persons(person_id),
                    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
                );
            """)

            # Table: stream_health (NEW: tracks live camera thread diagnostics)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stream_health (
                    camera_id TEXT PRIMARY KEY,
                    fps REAL NOT NULL,
                    last_frame_time TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reconnect_count INTEGER NOT NULL,
                    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE
                );
            """)

            # Table: recent_alerts (NEW: keeps tracks of alert times for cooldowns)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recent_alerts (
                    person_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    last_alert_time TEXT NOT NULL,
                    PRIMARY KEY (person_id, camera_id),
                    FOREIGN KEY (person_id) REFERENCES persons(person_id),
                    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
                );
            """)

            # Table: live_detections (NEW: records raw frame matches without cooldowns)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS live_detections (
                    detection_id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    similarity REAL NOT NULL,
                    detected_at TEXT NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES persons(person_id),
                    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
                );
            """)

            # Table: alert_rules (Phase 5)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_rules (
                    rule_id TEXT PRIMARY KEY,
                    camera_id TEXT, -- NULL maps to 'ALL' cameras
                    risk_level_threshold TEXT NOT NULL, -- Low, Medium, High, Critical
                    webhook_url TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1
                );
            """)

            # Table: alerts (Phase 5: Operational alert workflow states)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    status TEXT NOT NULL, -- ACTIVE, ACKNOWLEDGED, RESOLVED, FALSE_POSITIVE
                    severity_score REAL NOT NULL, -- Calculated numeric severity score
                    evidence_path TEXT,
                    assigned_operator TEXT,
                    resolution_notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE
                );
            """)

            # Table: alert_lifecycle (Phase 5: Alert state transitions audit trail)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_lifecycle (
                    lifecycle_id TEXT PRIMARY KEY,
                    alert_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    operator TEXT,
                    notes TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id) ON DELETE CASCADE
                );
            """)

            # Table: tracklets (Phase 6: Single-camera tracklet sessions)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tracklets (
                    tracklet_id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    person_id TEXT, -- Can be NULL or watchlist person_id / unknown_XXXX
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT NOT NULL, -- ACTIVE, EXPIRED
                    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id),
                    FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE SET NULL
                );
            """)

            # Table: reid_embeddings (Phase 6: ReID feature vectors for tracklets)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reid_embeddings (
                    reid_id TEXT PRIMARY KEY,
                    tracklet_id TEXT NOT NULL,
                    vector TEXT NOT NULL, -- JSON serialized normalized float array
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (tracklet_id) REFERENCES tracklets(tracklet_id) ON DELETE CASCADE
                );
            """)

            # Table: camera_movements (Phase 6: Movement graph camera transitions)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS camera_movements (
                    movement_id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL,
                    from_camera_id TEXT NOT NULL,
                    to_camera_id TEXT NOT NULL,
                    departure_time TEXT NOT NULL,
                    arrival_time TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    similarity REAL NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE,
                    FOREIGN KEY (from_camera_id) REFERENCES cameras(camera_id),
                    FOREIGN KEY (to_camera_id) REFERENCES cameras(camera_id)
                );
            """)

            # Table: surveillance_memory (Phase 7: RAG memory database table)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS surveillance_memory (
                    memory_id TEXT PRIMARY KEY,
                    reference_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    person_id TEXT,
                    camera_id TEXT,
                    timestamp TEXT NOT NULL,
                    document_text TEXT NOT NULL,
                    vector_id INTEGER NOT NULL UNIQUE,
                    evidence_path TEXT,
                    FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE
                );
            """)

            # Table: assistant_queries_audit (Phase 8: Assistant query logs audit trail)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS assistant_queries_audit (
                    query_id TEXT PRIMARY KEY,
                    user_username TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    retrieved_memory_ids TEXT NOT NULL, -- JSON string list of memory IDs
                    response_latency_ms REAL NOT NULL,
                    backend_used TEXT NOT NULL
                );
            """)

            # Performance Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_person_id ON events(person_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_face_embeddings_person_id ON face_embeddings(person_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_face_embeddings_vector_id ON face_embeddings(vector_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_live_detections_person_id ON live_detections(person_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_live_detections_detected_at ON live_detections(detected_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alert_lifecycle_alert_id ON alert_lifecycle(alert_id);")
            
            # Phase 6 Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tracklets_person_id ON tracklets(person_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tracklets_camera_id ON tracklets(camera_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reid_embeddings_tracklet_id ON reid_embeddings(tracklet_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_camera_movements_person_id ON camera_movements(person_id);")
            
            # Phase 7 Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_surveillance_memory_reference_id ON surveillance_memory(reference_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_surveillance_memory_person_id ON surveillance_memory(person_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_surveillance_memory_timestamp ON surveillance_memory(timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_surveillance_memory_vector_id ON surveillance_memory(vector_id);")
            
            # Phase 8 Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_assistant_queries_user ON assistant_queries_audit(user_username);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_assistant_queries_timestamp ON assistant_queries_audit(timestamp);")
            conn.commit()



    def register_camera(self, camera_id: str, location: str, video_source: str) -> bool:
        """Registers a camera or updates its location if it already exists."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO cameras (camera_id, location, video_source)
                    VALUES (?, ?, ?)
                    ON CONFLICT(camera_id) DO UPDATE SET
                        location=excluded.location,
                        video_source=excluded.video_source;
                """, (camera_id, location, video_source))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to register camera: {e}")
            return False

    def register_person(self, person_id: str, name: str, category: str, risk_level: str) -> bool:
        """Registers a person in the persons table."""
        created_at = datetime.utcnow().isoformat()
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO persons (person_id, name, category, risk_level, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(person_id) DO UPDATE SET
                        name=excluded.name,
                        category=excluded.category,
                        risk_level=excluded.risk_level;
                """, (person_id, name, category, risk_level, created_at))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to register person: {e}")
            return False

    def register_face_embedding(self, embedding_id: str, person_id: str, vector_id: int,
                                image_path: str, source_type: str, metadata: Dict[str, Any]) -> bool:
        """Registers an embedding vector association in the database."""
        created_at = datetime.utcnow().isoformat()
        metadata_str = json.dumps(metadata)
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO face_embeddings (embedding_id, person_id, vector_id, image_path, source_type, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (embedding_id, person_id, vector_id, image_path, source_type, created_at, metadata_str))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to register face embedding: {e}")
            return False

    def get_person(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a person and all their registered face embedding records."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT * FROM persons WHERE person_id = ?", (person_id,))
                person_row = cursor.fetchone()
                if not person_row:
                    return None
                
                person = dict(person_row)
                
                cursor = conn.execute("SELECT * FROM face_embeddings WHERE person_id = ?", (person_id,))
                embedding_rows = cursor.fetchall()
                
                embeddings = []
                for row in embedding_rows:
                    emb = dict(row)
                    emb["metadata"] = json.loads(emb["metadata"])
                    embeddings.append(emb)
                
                person["embeddings"] = embeddings
                return person
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to retrieve person details: {e}")
            return None

    def get_person_by_vector_id(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """Finds the corresponding person and embedding metadata for a FAISS index vector_id."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT p.*, f.embedding_id, f.image_path, f.source_type, f.metadata AS emb_metadata
                    FROM face_embeddings f
                    JOIN persons p ON f.person_id = p.person_id
                    WHERE f.vector_id = ?
                """, (vector_id,))
                row = cursor.fetchone()
                if row:
                    res = dict(row)
                    res["emb_metadata"] = json.loads(res["emb_metadata"])
                    return res
                return None
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to map vector_id: {e}")
            return None

    def log_search_audit(self, log_id: str, query_source: str, query_params: Dict[str, Any],
                         results_count: int, execution_time_ms: float) -> bool:
        """Logs a vector search query event for security auditing purposes."""
        query_timestamp = datetime.utcnow().isoformat()
        params_str = json.dumps(query_params)
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO search_audit_logs (log_id, query_timestamp, query_source, query_params, results_count, execution_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (log_id, query_timestamp, query_source, params_str, results_count, execution_time_ms))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to log search audit: {e}")
            return False

    def log_event(self, event_id: str, person_id: str, camera_id: str, video_source: str,
                  timestamp: str, frame_number: int, confidence: float,
                  bounding_box: List[int], match_details: Dict[str, Any]) -> bool:
        """Logs a face match detection event."""
        bbox_str = json.dumps(bounding_box)
        match_details_str = json.dumps(match_details)
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO events (event_id, person_id, camera_id, video_source, timestamp, frame_number, confidence, bounding_box, match_details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (event_id, person_id, camera_id, video_source, timestamp, frame_number, confidence, bbox_str, match_details_str))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to log event: {e}")
            return False

    def get_events_for_person(self, person_id: str) -> List[Dict[str, Any]]:
        """Retrieves all logged events for a given person, ordered chronologically by timestamp."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM events 
                    WHERE person_id = ? 
                    ORDER BY timestamp ASC
                """, (person_id,))
                rows = cursor.fetchall()
                events = []
                for row in rows:
                    evt = dict(row)
                    evt['bounding_box'] = json.loads(evt['bounding_box'])
                    evt['match_details'] = json.loads(evt['match_details'])
                    events.append(evt)
                return events
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to retrieve events: {e}")
            return []

    # --- Phase 4 Upgrades: Stream Health, Deduplication, Live Detections, Escalation ---
    
    def update_stream_health(self, camera_id: str, fps: float, status: str, reconnect_count: int) -> bool:
        """Inserts or updates stream diagnostic telemetry in SQLite stream_health table."""
        last_frame_time = datetime.utcnow().isoformat()
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO stream_health (camera_id, fps, last_frame_time, status, reconnect_count)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(camera_id) DO UPDATE SET
                        fps=excluded.fps,
                        last_frame_time=excluded.last_frame_time,
                        status=excluded.status,
                        reconnect_count=excluded.reconnect_count;
                """, (camera_id, fps, last_frame_time, status, reconnect_count))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to update stream health: {e}")
            return False

    def check_alert_cooldown(self, person_id: str, camera_id: str, cooldown_seconds: int) -> bool:
        """
        Evaluates and updates alert cooldown mappings.
        
        Returns:
            bool: True if cooldown is ACTIVE (throttle alert), False if INACTIVE (cooldown elapsed, fire alert).
        """
        now = datetime.utcnow()
        now_str = now.isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT last_alert_time FROM recent_alerts WHERE person_id = ? AND camera_id = ?;",
                    (person_id, camera_id)
                )
                row = cursor.fetchone()
                
                if row:
                    last_alert = datetime.fromisoformat(row[0])
                    time_diff = (now - last_alert).total_seconds()
                    
                    if time_diff < cooldown_seconds:
                        return True # Cooldown active
                    
                    # Cooldown elapsed, update timestamp
                    conn.execute(
                        "UPDATE recent_alerts SET last_alert_time = ? WHERE person_id = ? AND camera_id = ?;",
                        (now_str, person_id, camera_id)
                    )
                else:
                    # First match, insert new mapping
                    conn.execute(
                        "INSERT INTO recent_alerts (person_id, camera_id, last_alert_time) VALUES (?, ?, ?);",
                        (person_id, camera_id, now_str)
                    )
                conn.commit()
                return False
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Cooldown validation failed: {e}")
            return False

    def log_live_detection(self, detection_id: str, person_id: str, camera_id: str, similarity: float) -> bool:
        """Logs a raw frame face match to live_detections (runs for every matching frame)."""
        detected_at = datetime.utcnow().isoformat()
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO live_detections (detection_id, person_id, camera_id, similarity, detected_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (detection_id, person_id, camera_id, similarity, detected_at))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to log live detection: {e}")
            return False

    def check_multi_camera_sighting(self, person_id: str, exclude_camera_id: str, window_minutes: int) -> bool:
        """
        Checks if the person was detected on any camera OTHER than exclude_camera_id 
        within the past window_minutes. Used for alert priority escalation.
        """
        threshold_time = (datetime.utcnow() - timedelta(minutes=window_minutes)).isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM live_detections
                    WHERE person_id = ? AND camera_id != ? AND detected_at >= ?
                """, (person_id, exclude_camera_id, threshold_time))
                count = cursor.fetchone()[0]
                return count > 0
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Sighting escalation check failed: {e}")
            return False

    # --- Phase 5 Upgrades: Alert Rules and Alert Lifecycle Operations ---

    def create_alert_rule(self, rule_id: str, camera_id: Optional[str], risk_level_threshold: str, webhook_url: Optional[str], is_active: int = 1) -> bool:
        """Creates a dynamic alert rule in SQLite."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO alert_rules (rule_id, camera_id, risk_level_threshold, webhook_url, is_active)
                    VALUES (?, ?, ?, ?, ?);
                """, (rule_id, camera_id, risk_level_threshold, webhook_url, is_active))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to create alert rule: {e}")
            return False

    def get_alert_rules(self) -> List[Dict[str, Any]]:
        """Retrieves all active/inactive alert rules."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT * FROM alert_rules;")
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to retrieve alert rules: {e}")
            return []

    def get_alert_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single alert rule by ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT * FROM alert_rules WHERE rule_id = ?;", (rule_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to retrieve alert rule: {e}")
            return None

    def delete_alert_rule(self, rule_id: str) -> bool:
        """Deletes an alert rule."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM alert_rules WHERE rule_id = ?;", (rule_id,))
                conn.commit()
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to delete alert rule: {e}")
            return False

    def create_alert(self, alert_id: str, event_id: str, status: str, severity_score: float, evidence_path: Optional[str]) -> bool:
        """Creates an operational alert record and logs the initial transition status in lifecycle."""
        now_str = datetime.utcnow().isoformat()
        lifecycle_id = f"LC_{uuid.uuid4().hex[:8].upper()}"
        try:
            with self._get_connection() as conn:
                # 1. Insert alert
                conn.execute("""
                    INSERT INTO alerts (alert_id, event_id, status, severity_score, evidence_path, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                """, (alert_id, event_id, status, severity_score, evidence_path, now_str, now_str))
                
                # 2. Insert initial lifecycle record
                conn.execute("""
                    INSERT INTO alert_lifecycle (lifecycle_id, alert_id, status, operator, notes, timestamp)
                    VALUES (?, ?, ?, NULL, 'Alert initialized by system detector', ?);
                """, (lifecycle_id, alert_id, status, now_str))
                
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to create alert: {e}")
            return False

    def transition_alert_status(self, alert_id: str, new_status: str, operator: Optional[str], notes: Optional[str]) -> bool:
        """Updates alert status and inserts a record into the alert_lifecycle audit table."""
        now_str = datetime.utcnow().isoformat()
        lifecycle_id = f"LC_{uuid.uuid4().hex[:8].upper()}"
        try:
            with self._get_connection() as conn:
                # 1. Update alert
                cursor = conn.execute("""
                    UPDATE alerts 
                    SET status = ?, assigned_operator = ?, resolution_notes = ?, updated_at = ?
                    WHERE alert_id = ?;
                """, (new_status, operator, notes, now_str, alert_id))
                
                if cursor.rowcount == 0:
                    return False
                    
                # 2. Insert lifecycle audit trail
                conn.execute("""
                    INSERT INTO alert_lifecycle (lifecycle_id, alert_id, status, operator, notes, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?);
                """, (lifecycle_id, alert_id, new_status, operator, notes, now_str))
                
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to transition alert status: {e}")
            return False

    def get_alerts(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves operational alerts, optionally filtered by status, joined with event details."""
        try:
            with self._get_connection() as conn:
                if status:
                    cursor = conn.execute("""
                        SELECT a.*, e.person_id, e.camera_id, e.timestamp AS sighting_time, p.name AS person_name
                        FROM alerts a
                        JOIN events e ON a.event_id = e.event_id
                        LEFT JOIN persons p ON e.person_id = p.person_id
                        WHERE a.status = ?
                        ORDER BY a.created_at DESC
                        LIMIT ?;
                    """, (status, limit))
                else:
                    cursor = conn.execute("""
                        SELECT a.*, e.person_id, e.camera_id, e.timestamp AS sighting_time, p.name AS person_name
                        FROM alerts a
                        JOIN events e ON a.event_id = e.event_id
                        LEFT JOIN persons p ON e.person_id = p.person_id
                        ORDER BY a.created_at DESC
                        LIMIT ?;
                    """, (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to retrieve alerts: {e}")
            return []

    def get_alert_with_event(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single alert joined with event details."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT a.*, e.person_id, e.camera_id, e.timestamp AS sighting_time, e.bounding_box, e.confidence, p.name AS person_name
                    FROM alerts a
                    JOIN events e ON a.event_id = e.event_id
                    LEFT JOIN persons p ON e.person_id = p.person_id
                    WHERE a.alert_id = ?;
                """, (alert_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to retrieve alert with event: {e}")
            return None

    def get_alert_lifecycle_timeline(self, alert_id: str) -> List[Dict[str, Any]]:
        """Retrieves chronological lifecycle transitions for a specific alert."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT status, operator, notes, timestamp 
                    FROM alert_lifecycle 
                    WHERE alert_id = ? 
                    ORDER BY timestamp ASC;
                """, (alert_id,))
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to retrieve alert lifecycle timeline: {e}")
            return []

    def get_false_positive_analytics(self) -> Dict[str, Any]:
        """Calculates false positive rates and aggregates false positive counts by camera and person."""
        try:
            with self._get_connection() as conn:
                # Total counts by status
                cursor = conn.execute("SELECT status, COUNT(*) FROM alerts GROUP BY status;")
                counts = {row[0]: row[1] for row in cursor.fetchall()}
                
                total_alerts = sum(counts.values())
                active_count = counts.get("ACTIVE", 0)
                acknowledged_count = counts.get("ACKNOWLEDGED", 0)
                resolved_count = counts.get("RESOLVED", 0)
                fp_count = counts.get("FALSE_POSITIVE", 0)
                
                total_closed = resolved_count + fp_count
                fp_rate = fp_count / total_closed if total_closed > 0 else 0.0
                
                # False positives by camera
                cursor = conn.execute("""
                    SELECT e.camera_id, c.location, COUNT(*)
                    FROM alerts a
                    JOIN events e ON a.event_id = e.event_id
                    LEFT JOIN cameras c ON e.camera_id = c.camera_id
                    WHERE a.status = 'FALSE_POSITIVE'
                    GROUP BY e.camera_id;
                """)
                fp_by_camera = [
                    {"camera_id": row[0], "location": row[1] or "Unknown", "count": row[2]}
                    for row in cursor.fetchall()
                ]
                
                # False positives by person
                cursor = conn.execute("""
                    SELECT e.person_id, p.name, COUNT(*)
                    FROM alerts a
                    JOIN events e ON a.event_id = e.event_id
                    LEFT JOIN persons p ON e.person_id = p.person_id
                    WHERE a.status = 'FALSE_POSITIVE'
                    GROUP BY e.person_id;
                """)
                fp_by_person = [
                    {"person_id": row[0], "name": row[1] or "Unknown", "count": row[2]}
                    for row in cursor.fetchall()
                ]
                
                return {
                    "total_alerts": total_alerts,
                    "active_count": active_count,
                    "acknowledged_count": acknowledged_count,
                    "resolved_count": resolved_count,
                    "false_positive_count": fp_count,
                    "false_positive_rate": round(fp_rate, 4),
                    "false_positives_by_camera": fp_by_camera,
                    "false_positives_by_person": fp_by_person
                }
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to fetch false positive analytics: {e}")
            return {}

    def archive_old_alerts(self, retention_days: int) -> int:
        """
        Deletes resolved and false-positive alerts older than retention_days.
        Preserves the events and audit logs.
        Also purges corresponding RAG memories.
        """
        from datetime import datetime, timedelta
        threshold_time = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()
        try:
            with self._get_connection() as conn:
                # 1. Fetch alert IDs to be deleted
                cursor = conn.execute("""
                    SELECT alert_id FROM alerts
                    WHERE (status = 'RESOLVED' OR status = 'FALSE_POSITIVE')
                      AND created_at < ?;
                """, (threshold_time,))
                alert_ids = [row[0] for row in cursor.fetchall()]
                
                if not alert_ids:
                    return 0
                
                # 2. Purge corresponding RAG memory documents
                placeholders = ",".join("?" for _ in alert_ids)
                conn.execute(f"""
                    DELETE FROM surveillance_memory
                    WHERE reference_id IN ({placeholders})
                """, alert_ids)
                
                # 3. Delete alerts
                cursor = conn.execute(f"""
                    DELETE FROM alerts
                    WHERE alert_id IN ({placeholders})
                """, alert_ids)
                
                conn.commit()
                return cursor.rowcount
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to archive old alerts and memories: {e}")
            return 0


    # --- Phase 6: Multi-Camera Tracking Helper Methods ---

    def create_tracklet(self, tracklet_id: str, camera_id: str, start_time: str, person_id: Optional[str] = None) -> bool:
        """Registers a new single-camera tracklet session."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO tracklets (tracklet_id, camera_id, person_id, start_time, status)
                    VALUES (?, ?, ?, ?, 'ACTIVE')
                """, (tracklet_id, camera_id, person_id, start_time))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to create tracklet: {e}")
            return False

    def update_tracklet_person(self, tracklet_id: str, person_id: str) -> bool:
        """Associates a tracklet with an identified person (watchlist or unknown)."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE tracklets
                    SET person_id = ?
                    WHERE tracklet_id = ?
                """, (person_id, tracklet_id))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to update tracklet person: {e}")
            return False

    def close_tracklet(self, tracklet_id: str, end_time: str) -> bool:
        """Closes a tracklet session, setting its status to EXPIRED."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE tracklets
                    SET end_time = ?, status = 'EXPIRED'
                    WHERE tracklet_id = ?
                """, (end_time, tracklet_id))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to close tracklet: {e}")
            return False

    def get_active_tracklets(self, camera_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves currently active tracklets, optionally filtered by camera."""
        try:
            with self._get_connection() as conn:
                if camera_id:
                    cursor = conn.execute("""
                        SELECT t.*, p.name AS person_name
                        FROM tracklets t
                        LEFT JOIN persons p ON t.person_id = p.person_id
                        WHERE t.status = 'ACTIVE' AND t.camera_id = ?
                    """, (camera_id,))
                else:
                    cursor = conn.execute("""
                        SELECT t.*, p.name AS person_name
                        FROM tracklets t
                        LEFT JOIN persons p ON t.person_id = p.person_id
                        WHERE t.status = 'ACTIVE'
                    """)
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to fetch active tracklets: {e}")
            return []

    def register_reid_embedding(self, reid_id: str, tracklet_id: str, vector: List[float]) -> bool:
        """Registers a 256-D body ReID feature vector for a tracklet."""
        created_at = datetime.utcnow().isoformat()
        vector_str = json.dumps(vector)
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO reid_embeddings (reid_id, tracklet_id, vector, created_at)
                    VALUES (?, ?, ?, ?)
                """, (reid_id, tracklet_id, vector_str, created_at))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to register ReID embedding: {e}")
            return False

    def get_recent_reid_embeddings(self, limit_minutes: float = 10.0) -> List[Dict[str, Any]]:
        """Retrieves ReID embeddings from tracklets active or modified recently."""
        threshold_time = (datetime.utcnow() - timedelta(minutes=limit_minutes)).isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT r.reid_id, r.tracklet_id, r.vector, r.created_at, t.camera_id, t.person_id
                    FROM reid_embeddings r
                    JOIN tracklets t ON r.tracklet_id = t.tracklet_id
                    WHERE r.created_at >= ?
                """, (threshold_time,))
                results = []
                for row in cursor.fetchall():
                    res = dict(row)
                    res["vector"] = json.loads(res["vector"])
                    results.append(res)
                return results
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to fetch recent ReID embeddings: {e}")
            return []

    def register_movement(self, person_id: str, from_camera_id: str, to_camera_id: str,
                          departure_time: str, arrival_time: str, duration_seconds: float, similarity: float,
                          movement_id: Optional[str] = None) -> Optional[str]:
        """Registers a cross-camera movement transition (for the Movement Graph)."""
        if not movement_id:
            movement_id = f"MVT_{uuid.uuid4().hex[:8].upper()}"
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO camera_movements (movement_id, person_id, from_camera_id, to_camera_id, departure_time, arrival_time, duration_seconds, similarity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (movement_id, person_id, from_camera_id, to_camera_id, departure_time, arrival_time, duration_seconds, similarity))
                conn.commit()
                return movement_id
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to register movement: {e}")
            return None


    def get_movements(self, person_id: str) -> List[Dict[str, Any]]:
        """Retrieves chronological camera transition movements for a person."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT m.*, c1.location AS from_location, c2.location AS to_location
                    FROM camera_movements m
                    JOIN cameras c1 ON m.from_camera_id = c1.camera_id
                    JOIN cameras c2 ON m.to_camera_id = c2.camera_id
                    WHERE m.person_id = ?
                    ORDER BY m.arrival_time ASC
                """, (person_id,))
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to fetch movements: {e}")
            return []

    def get_camera_visit_counts(self) -> List[Dict[str, Any]]:
        """Fetches camera sighting visit counts to prepare for heatmaps."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT c.camera_id, c.location, COUNT(e.event_id) AS visit_count
                    FROM cameras c
                    LEFT JOIN events e ON c.camera_id = e.camera_id
                    GROUP BY c.camera_id
                    ORDER BY visit_count DESC
                """)
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to fetch camera visits: {e}")
            return []

    # --- Phase 7: RAG Surveillance Memory Database Methods ---

    def log_memory_document(self, memory_id: str, reference_id: str, entity_type: str,
                            person_id: Optional[str], camera_id: Optional[str],
                            timestamp: str, document_text: str, vector_id: int,
                            evidence_path: Optional[str] = None) -> bool:
        """Registers a natural language memory document in the database."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO surveillance_memory (memory_id, reference_id, entity_type, person_id, camera_id, timestamp, document_text, vector_id, evidence_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (memory_id, reference_id, entity_type, person_id, camera_id, timestamp, document_text, vector_id, evidence_path))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to log memory document: {e}")
            return False

    def delete_memory_by_reference(self, reference_id: str) -> bool:
        """Deletes memory documents associated with a reference ID."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM surveillance_memory WHERE reference_id = ?", (reference_id,))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to delete memory document: {e}")
            return False

    def get_memory_by_vector_id(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """Maps a vector ID back to its corresponding memory text and metadata."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT * FROM surveillance_memory WHERE vector_id = ?", (vector_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to map memory vector ID: {e}")
            return None

    def get_memories(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetches all registered memory documents from the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT * FROM surveillance_memory ORDER BY timestamp DESC LIMIT ?", (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to fetch memories: {e}")
            return []

    def log_assistant_query(self, user_username: str, query: str, retrieved_memory_ids: List[str],
                            latency_ms: float, backend_used: str) -> bool:
        """Logs an assistant query session to the audit database for security compliance."""
        query_id = f"ASQ_{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.utcnow().isoformat()
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO assistant_queries_audit (query_id, user_username, timestamp, query, retrieved_memory_ids, response_latency_ms, backend_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (query_id, user_username, timestamp, query, json.dumps(retrieved_memory_ids), latency_ms, backend_used))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[DATABASE ERROR] Failed to log assistant query audit trail: {e}")
            return False



