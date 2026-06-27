import cv2
import numpy as np
import threading
import time
import uuid
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from fastapi import WebSocket

from config import settings
from core.detector import FaceDetectorYuNet
from core.recognizer import FaceRecognizerSFace
from core.event_engine import EventEngine
from core.vector_db import VectorDBManager
from core.pipeline import SurveillancePipeline
from core.alert_engine import AlertCoordinator
from core.tracker import SingleCameraTracker, Tracklet
from core.reid import PersonReIDExtractor, YoloPersonDetector, estimate_body_box
from core.rag_memory import (
    SurveillanceMemoryManager,
    formulate_sighting_text,
    formulate_alert_text,
    formulate_tracklet_text,
    formulate_movement_text
)



# --- WebSocket Broadcast Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = threading.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self._lock:
            self.active_connections.append(websocket)
        print(f"[WS] Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        print(f"[WS] Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Sends a JSON payload to all active WebSocket connections in parallel."""
        disconnected_clients = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # Mark failed connections for removal
                disconnected_clients.append(connection)
                
        for client in disconnected_clients:
            self.disconnect(client)

# Instantiate Global WebSocket Manager
ws_manager = ConnectionManager()


# --- Live CCTV Stream Worker ---
class LiveStreamWorker:
    def __init__(self, camera_id: str, location: str, stream_source: str,
                 threshold: float, detector: FaceDetectorYuNet, recognizer: FaceRecognizerSFace,
                 event_engine: EventEngine, vector_db: VectorDBManager, ws_broadcaster: ConnectionManager):
        self.camera_id = camera_id
        self.location = location
        self.stream_source = stream_source
        self.threshold = threshold
        
        self.detector = detector
        self.recognizer = recognizer
        self.event_engine = event_engine
        self.vector_db = vector_db
        self.ws_broadcaster = ws_broadcaster
        self.alert_coordinator = AlertCoordinator(self.event_engine)

        # Multi-Camera Person Tracking Instantiations
        self.tracker = SingleCameraTracker(camera_id=self.camera_id, timeout_seconds=settings.TRACKLET_TIMEOUT)
        self.reid_extractor = PersonReIDExtractor(settings.REID_MODEL_PATH)
        self.rag_memory = SurveillanceMemoryManager(self.event_engine)
        
        self.yolo_detector = None
        if settings.TRACKING_DETECTION_MODE == "yolo":
            self.yolo_detector = YoloPersonDetector(settings.YOLO_MODEL_PATH)

        self._running = False


        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        
        self._reader_thread: Optional[threading.Thread] = None
        self._analyzer_thread: Optional[threading.Thread] = None
        
        self.reconnect_count = 0
        self.current_fps = 0.0

    def start(self):
        """Launches background threads for the zero-lag frame reader and analyzer loops."""
        if self._running:
            return
        
        # Ensure camera metadata is registered
        self.event_engine.register_camera(self.camera_id, self.location, self.stream_source)
        
        self._running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._analyzer_thread = threading.Thread(target=self._analyzer_loop, daemon=True)
        
        self._reader_thread.start()
        self._analyzer_thread.start()
        print(f"[STREAM WORKER] Started live monitoring on Camera {self.camera_id} ({self.location})")

    def stop(self):
        """Stops background threads and releases stream resources."""
        self._running = False
        if self._reader_thread:
            self._reader_thread.join(timeout=2.0)
        if self._analyzer_thread:
            self._analyzer_thread.join(timeout=2.0)
        
        # Log offline status
        self.event_engine.update_stream_health(self.camera_id, 0.0, "OFFLINE", self.reconnect_count)
        print(f"[STREAM WORKER] Stopped live monitoring on Camera {self.camera_id}")

    def _reader_loop(self):
        """Reader Thread: Constantly grabs frames into a shared single-frame buffer."""
        cap = cv2.VideoCapture(self.stream_source)
        
        # Update initial stream health to ONLINE
        self.event_engine.update_stream_health(self.camera_id, 30.0, "ONLINE", self.reconnect_count)

        fps_timer = time.perf_counter()
        frame_counter = 0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                print(f"[STREAM WORKER] Connection lost on Camera {self.camera_id}. Initiating reconnection...")
                cap.release()
                
                # Update status to RECONNECTING
                self.event_engine.update_stream_health(self.camera_id, 0.0, "RECONNECTING", self.reconnect_count)
                
                reconnected = False
                for attempt in range(1, settings.MAX_RECONNECT_ATTEMPTS + 1):
                    if not self._running:
                        break
                    print(f"[STREAM RECOVER] Reconnect attempt {attempt}/{settings.MAX_RECONNECT_ATTEMPTS} on Camera {self.camera_id}...")
                    time.sleep(5)
                    cap = cv2.VideoCapture(self.stream_source)
                    if cap.isOpened():
                        print(f"[STREAM RECOVER] Reconnect SUCCESS on Camera {self.camera_id}!")
                        self.reconnect_count += 1
                        self.event_engine.update_stream_health(self.camera_id, 30.0, "ONLINE", self.reconnect_count)
                        reconnected = True
                        break
                
                if not reconnected:
                    print(f"[STREAM WORKER] Camera {self.camera_id} is OFFLINE. Reconnect failed.")
                    self.event_engine.update_stream_health(self.camera_id, 0.0, "OFFLINE", self.reconnect_count)
                    self._running = False
                    break
                continue

            frame_counter += 1
            # Update local FPS calculations every 30 frames
            if frame_counter % 30 == 0:
                elapsed = time.perf_counter() - fps_timer
                self.current_fps = 30.0 / elapsed if elapsed > 0 else 0.0
                fps_timer = time.perf_counter()
                # Update diagnostics database health
                self.event_engine.update_stream_health(self.camera_id, round(self.current_fps, 2), "ONLINE", self.reconnect_count)

            # Thread-safe copy to shared buffer (discards older frame if busy)
            with self._frame_lock:
                self._latest_frame = frame.copy()

            # Small sleep to prevent tight loop throttling CPU cores
            time.sleep(0.01)

        cap.release()

    def _analyzer_loop(self):
        """Analyzer Thread: Pulls the newest frame from the shared buffer, runs tracking, ReID, and alert dispatches."""
        while self._running:
            frame = None
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()
                    self._latest_frame = None  # Consume the frame
            
            if frame is None:
                # Buffer is empty, wait for next frame
                time.sleep(0.01)
                continue

            # Process frame
            h, w = frame.shape[:2]
            det_frame = frame
            scale_x, scale_y = 1.0, 1.0
            if settings.RESIZE_WIDTH and w > settings.RESIZE_WIDTH:
                new_w = settings.RESIZE_WIDTH
                new_h = int(h * (new_w / w))
                det_frame = cv2.resize(frame, (new_w, new_h))
                scale_x = w / new_w
                scale_y = h / new_h

            detections_for_tracker = []

            # 1. Detect faces using YuNet
            face_detections = self.detector.detect(det_frame)

            # 2. Extract facial recognition details for each face
            face_matches = []
            for face_det in face_detections:
                box = face_det["box"]
                native_box = [
                    int(box[0] * scale_x),
                    int(box[1] * scale_y),
                    int(box[2] * scale_x),
                    int(box[3] * scale_y)
                ]
                
                # Align and embed
                aligned = self.recognizer.align_face(det_frame, face_det["raw"])
                embedding = self.recognizer.extract_embedding(aligned)

                # Search FAISS index
                matches = self.vector_db.search(
                    embedding=embedding,
                    k=1,
                    threshold=self.threshold,
                    query_source="live_surveillance"
                )
                
                match = matches[0] if matches else None
                face_matches.append({
                    "face_box": native_box,
                    "match": match,
                    "raw_det": face_det["raw"]
                })

            # 3. Determine Mode and associate boxes
            if settings.TRACKING_DETECTION_MODE == "yolo" and self.yolo_detector is not None:
                # Mode 2: YOLO Person Detection
                yolo_detections = self.yolo_detector.detect(frame)
                for person in yolo_detections:
                    body_box = person["box"]
                    bx, by, bw, bh = body_box
                    
                    # Crop body and extract ReID
                    body_crop = frame[max(0, by):min(h, by+bh), max(0, bx):min(w, bx+bw)]
                    reid_vector = self.reid_extractor.extract_embedding(body_crop)
                    
                    # Link face if center of face lies within body box
                    linked_person_id = None
                    linked_match = None
                    linked_raw_det = None
                    for fm in face_matches:
                        fbx, fby, fbw, fbh = fm["face_box"]
                        fcx = fbx + fbw / 2
                        fcy = fby + fbh / 2
                        if bx <= fcx <= bx + bw and by <= fcy <= by + bh:
                            if fm["match"]:
                                linked_person_id = fm["match"]["person_id"]
                                linked_match = fm["match"]
                                linked_raw_det = fm["raw_det"]
                            break
                            
                    detections_for_tracker.append({
                        "box": body_box,
                        "body_box": body_box,
                        "reid_embedding": reid_vector.tolist(),
                        "person_id": linked_person_id,
                        "face_match": linked_match,
                        "raw_det": linked_raw_det
                    })
            else:
                # Mode 1: Face-Guided Body Crop Estimation (Default)
                for fm in face_matches:
                    face_box = fm["face_box"]
                    body_box = estimate_body_box(face_box, frame.shape)
                    
                    # Crop estimated body and extract ReID
                    bx, by, bw, bh = body_box
                    body_crop = frame[max(0, by):min(h, by+bh), max(0, bx):min(w, bx+bw)]
                    reid_vector = self.reid_extractor.extract_embedding(body_crop)
                    
                    person_id = fm["match"]["person_id"] if fm["match"] else None
                    detections_for_tracker.append({
                        "box": face_box,  # Face bounding box is tracked
                        "body_box": body_box,
                        "reid_embedding": reid_vector.tolist(),
                        "person_id": person_id,
                        "face_match": fm["match"],
                        "raw_det": fm["raw_det"]
                    })

            # 4. Update Single Camera Tracker
            active_tracklet_matches, expired_tracklets = self.tracker.update(detections_for_tracker)

            # 5. Process active tracklets
            start_time_str = datetime.utcnow().isoformat()
            
            for tracklet, det in active_tracklet_matches:
                tracklet.add_reid_embedding(det["reid_embedding"])
                
                # Check if first sighting of this tracklet
                if not tracklet.has_logged_db:
                    self.event_engine.create_tracklet(tracklet.tracklet_id, self.camera_id, start_time_str, tracklet.person_id)
                    tracklet.has_logged_db = True
                    self._update_tracklet_memory(tracklet, start_time_str)

                # If tracklet does not have a person identity, check for matches
                if tracklet.person_id is None:
                    # 5.1 Try Face Match
                    if det["person_id"] is not None:
                        tracklet.person_id = det["person_id"]
                        self.event_engine.update_tracklet_person(tracklet.tracklet_id, tracklet.person_id)
                        self._update_tracklet_memory(tracklet, start_time_str)
                    # 5.2 Try Cross-Camera ReID Match
                    else:
                        recent_embs = self.event_engine.get_recent_reid_embeddings(limit_minutes=10.0)
                        other_embs = [r for r in recent_embs if r["camera_id"] != self.camera_id]
                        if other_embs:
                            # Compare current vector against all other camera vectors
                            curr_vec = np.array(det["reid_embedding"])
                            best_sim = -1.0
                            best_emb = None
                            for other in other_embs:
                                sim = float(np.dot(curr_vec, np.array(other["vector"])))
                                if sim > best_sim:
                                    best_sim = sim
                                    best_emb = other
                                    
                            if best_sim >= settings.REID_THRESHOLD and best_emb is not None:
                                matched_pid = best_emb["person_id"]
                                if matched_pid:
                                    tracklet.person_id = matched_pid
                                    self.event_engine.update_tracklet_person(tracklet.tracklet_id, matched_pid)
                                    self._update_tracklet_memory(tracklet, start_time_str)
                                    
                                    # Log Camera Transition Movement
                                    # Departure time is when the match was recorded, arrival time is now
                                    departure_time = best_emb["created_at"]
                                    arrival_time = datetime.utcnow().isoformat()
                                    try:
                                        t_dep = datetime.fromisoformat(departure_time)
                                        t_arr = datetime.fromisoformat(arrival_time)
                                        duration = (t_arr - t_dep).total_seconds()
                                    except Exception:
                                        duration = 0.0
                                    
                                    mvt_id = self.event_engine.register_movement(
                                        person_id=matched_pid,
                                        from_camera_id=best_emb["camera_id"],
                                        to_camera_id=self.camera_id,
                                        departure_time=departure_time,
                                        arrival_time=arrival_time,
                                        duration_seconds=duration,
                                        similarity=best_sim
                                    )
                                    if mvt_id:
                                        # Log RAG transition memory document
                                        p_info = self.event_engine.get_person(matched_pid)
                                        p_name = p_info["name"] if p_info else "Target"
                                        from_loc = "Unknown Location"
                                        try:
                                            with self.event_engine._get_connection() as conn:
                                                cursor = conn.execute("SELECT location FROM cameras WHERE camera_id = ?", (best_emb["camera_id"],))
                                                row = cursor.fetchone()
                                                if row:
                                                    from_loc = row[0]
                                        except Exception:
                                            pass
                                            
                                        mvt_txt = formulate_movement_text(
                                            name=p_name,
                                            person_id=matched_pid,
                                            from_loc=from_loc,
                                            from_cam=best_emb["camera_id"],
                                            to_loc=self.location,
                                            to_cam=self.camera_id,
                                            departure_time=departure_time,
                                            arrival_time=arrival_time,
                                            duration=duration,
                                            similarity=best_sim
                                        )
                                        self.rag_memory.add_memory(
                                            reference_id=mvt_id,
                                            entity_type="movement",
                                            person_id=matched_pid,
                                            camera_id=self.camera_id,
                                            timestamp=arrival_time,
                                            document_text=mvt_txt
                                        )
                                    print(f"[REID PROPAGATION] Target matches {matched_pid} across cameras (similarity {best_sim:.2f})")
                                    
                    # 5.3 If STILL None, register a new Unknown Target
                    if tracklet.person_id is None:
                        unknown_id = f"unknown_{uuid.uuid4().hex[:6].upper()}"
                        self.event_engine.register_person(
                            person_id=unknown_id,
                            name=f"Unknown Target {unknown_id.split('_')[1]}",
                            category="Unknown",
                            risk_level="Low"
                        )
                        tracklet.person_id = unknown_id
                        self.event_engine.update_tracklet_person(tracklet.tracklet_id, unknown_id)
                        self._update_tracklet_memory(tracklet, start_time_str)


                # 5.4 Register current frame's ReID feature vector
                reid_id = f"REID_{uuid.uuid4().hex[:8].upper()}"
                self.event_engine.register_reid_embedding(reid_id, tracklet.tracklet_id, det["reid_embedding"])

                # 5.5 If target is a watchlist target, log sighting event and trigger Alert Engine (Phase 5)
                if tracklet.person_id and not tracklet.person_id.startswith("unknown_"):
                    person_details = self.event_engine.get_person(tracklet.person_id)
                    if person_details and person_details.get("category") != "Unknown":
                        base_risk_level = person_details.get("risk_level", "Low")
                        name = person_details.get("name", "Watchlist Target")
                        
                        # Use similarity from face match if available, else default to threshold
                        sim_score = det["face_match"]["similarity"] if det["face_match"] else settings.REID_THRESHOLD
                        
                        # Cooldown check
                        cooldown_active = self.event_engine.check_alert_cooldown(
                            person_id=tracklet.person_id,
                            camera_id=self.camera_id,
                            cooldown_seconds=settings.ALERT_COOLDOWN_SECONDS
                        )
                        
                        if not cooldown_active:
                            # Sighting Escalation
                            final_risk_level = base_risk_level
                            escalated = self.event_engine.check_multi_camera_sighting(
                                person_id=tracklet.person_id,
                                exclude_camera_id=self.camera_id,
                                window_minutes=settings.ESCALATION_TIME_WINDOW_MINUTES
                            )
                            if escalated:
                                final_risk_level = "Critical"
                                print(f"[ESCALATE] Escalating match for {name} to CRITICAL (seen on multiple cameras).")

                            # Save evidence files
                            alert_id = f"ALT_{uuid.uuid4().hex[:8].upper()}"
                            evidence_path = self._save_evidence_files(alert_id, frame, det["box"], name, final_risk_level, sim_score)

                            # Log event
                            timestamp_str = datetime.utcnow().isoformat()
                            match_details = {
                                "match": True,
                                "confidence": sim_score,
                                "threshold": self.threshold,
                                "risk_level": final_risk_level,
                                "escalated": escalated,
                                "embedding_id": det["face_match"]["embedding_id"] if det["face_match"] else "reid_propagation",
                                "evidence_path": str(evidence_path)
                            }
                            
                            self.event_engine.log_event(
                                event_id=alert_id,
                                person_id=tracklet.person_id,
                                camera_id=self.camera_id,
                                video_source="LIVE_STREAM",
                                timestamp=timestamp_str,
                                frame_number=-1,
                                confidence=sim_score,
                                bounding_box=det["box"],
                                match_details=match_details
                            )

                            # Log RAG Sighting Event Memory
                            sighting_txt = formulate_sighting_text(
                                name=name,
                                person_id=tracklet.person_id,
                                camera_location=self.location,
                                camera_id=self.camera_id,
                                timestamp=timestamp_str,
                                similarity=sim_score
                            )
                            self.rag_memory.add_memory(
                                reference_id=alert_id,
                                entity_type="event",
                                person_id=tracklet.person_id,
                                camera_id=self.camera_id,
                                timestamp=timestamp_str,
                                document_text=sighting_txt,
                                evidence_path=str(evidence_path)
                            )

                            # Evaluate operational Alert Rules and dispatch notifications (Phase 5)
                            alert_triggered = self.alert_coordinator.evaluate_rules_and_dispatch(
                                alert_id=alert_id,
                                person_id=tracklet.person_id,
                                name=name,
                                risk_level=final_risk_level,
                                camera_id=self.camera_id,
                                location=self.location,
                                similarity=sim_score,
                                escalated=escalated,
                                evidence_path=evidence_path
                            )

                            if alert_triggered:
                                severity_score = self.alert_coordinator.calculate_severity_score(final_risk_level, sim_score, escalated)
                                alert_txt = formulate_alert_text(
                                    name=name,
                                    person_id=tracklet.person_id,
                                    camera_location=self.location,
                                    camera_id=self.camera_id,
                                    timestamp=timestamp_str,
                                    severity=severity_score,
                                    status="ACTIVE"
                                )
                                self.rag_memory.add_memory(
                                    reference_id=alert_id,
                                    entity_type="alert",
                                    person_id=tracklet.person_id,
                                    camera_id=self.camera_id,
                                    timestamp=timestamp_str,
                                    document_text=alert_txt,
                                    evidence_path=str(evidence_path)
                                )


                            # Broadcast Real-Time JSON Alert to Connected WebSockets
                            alert_payload = {
                                "alert_type": "watchlist_match",
                                "alert_id": alert_id,
                                "person_id": tracklet.person_id,
                                "name": name,
                                "risk_level": final_risk_level,
                                "camera_id": self.camera_id,
                                "location": self.location,
                                "timestamp": timestamp_str,
                                "similarity": round(sim_score, 2),
                                "escalated": escalated,
                                "crop_image_url": f"/evidence/{self.camera_id}/{alert_id}_crop.jpg"
                            }
                            import asyncio
                            asyncio.run(self.ws_broadcaster.broadcast(alert_payload))

            # 6. Process expired tracklets to close their session
            for exp_tracklet in expired_tracklets:
                close_time = datetime.utcnow().isoformat()
                self.event_engine.close_tracklet(exp_tracklet.tracklet_id, close_time)
                
                # Convert start_time float timestamp to ISO string
                start_iso = datetime.utcfromtimestamp(exp_tracklet.start_time).isoformat() + "Z"
                exp_tracklet.end_time = close_time
                self._update_tracklet_memory(exp_tracklet, start_iso)


            # Rest the thread slightly
            time.sleep(0.03)

    def _save_evidence_files(self, alert_id: str, frame: np.ndarray, bbox: List[int],
                             name: str, risk_level: str, similarity: float) -> Path:
        """Saves crop, raw snapshot, and annotated frames inside a structured evidence directory."""
        now = datetime.now()
        # Path: evidence/<camera_id>/<YYYY>/<MM>/<DD>/
        relative_dir = Path(self.camera_id) / f"{now.year:04d}" / f"{now.month:02d}" / f"{now.day:02d}"
        target_dir = settings.EVIDENCE_DIR / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        h, w = frame.shape[:2]
        x, y, box_w, box_h = bbox
        x, y = max(0, x), max(0, y)
        box_w = min(w - x, box_w)
        box_h = min(h - y, box_h)

        # 1. Save Crop
        if box_w > 0 and box_h > 0:
            crop = frame[y:y+box_h, x:x+box_w]
            cv2.imwrite(str(target_dir / f"{alert_id}_crop.jpg"), crop)

        # 2. Save Native Snapshot
        cv2.imwrite(str(target_dir / f"{alert_id}_frame.jpg"), frame)

        # 3. Save Annotated Frame
        annotated = frame.copy()
        color = (0, 0, 255) if risk_level in ["Critical", "High"] else (0, 255, 0)
        cv2.rectangle(annotated, (x, y), (x + box_w, y + box_h), color, 2)
        label = f"{name} ({similarity:.2f}) - {risk_level}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (x, y - label_h - 10), (x + label_w, y), color, -1)
        cv2.putText(annotated, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        
        cv2.imwrite(str(target_dir / f"{alert_id}_annotated.jpg"), annotated)

        return target_dir

    def _update_tracklet_memory(self, tracklet: Tracklet, start_time_str: str):
        """Creates or updates the RAG memory document for a tracklet session."""
        name = "Unknown Target"
        if tracklet.person_id:
            person_details = self.event_engine.get_person(tracklet.person_id)
            if person_details:
                name = person_details.get("name", "Unknown Target")
        
        end_time_val = getattr(tracklet, "end_time", None)
        status_str = "ACTIVE" if not end_time_val else "EXPIRED"
        txt = formulate_tracklet_text(
            name=name,
            person_id=tracklet.person_id or "unregistered",
            camera_location=self.location,
            camera_id=self.camera_id,
            start_time=start_time_str,
            end_time=end_time_val,
            status=status_str
        )
        self.rag_memory.add_memory(
            reference_id=tracklet.tracklet_id,
            entity_type="tracklet",
            person_id=tracklet.person_id,
            camera_id=self.camera_id,
            timestamp=start_time_str,
            document_text=txt,
            evidence_path=None
        )



# --- Surveillance Camera Registry ---
class StreamRegistry:
    def __init__(self, detector: FaceDetectorYuNet, recognizer: FaceRecognizerSFace,
                 event_engine: EventEngine, vector_db: VectorDBManager, ws_broadcaster: ConnectionManager):
        self.detector = detector
        self.recognizer = recognizer
        self.event_engine = event_engine
        self.vector_db = vector_db
        self.ws_broadcaster = ws_broadcaster
        
        self.workers: Dict[str, LiveStreamWorker] = {}
        self._lock = threading.Lock()

    def start_stream(self, camera_id: str, location: str, stream_source: str, threshold: float = 0.40) -> bool:
        """Spawns and starts a stream monitoring worker thread."""
        with self._lock:
            if camera_id in self.workers:
                print(f"[REGISTRY] Camera {camera_id} is already running.")
                return False

            worker = LiveStreamWorker(
                camera_id=camera_id,
                location=location,
                stream_source=stream_source,
                threshold=threshold,
                detector=self.detector,
                recognizer=self.recognizer,
                event_engine=self.event_engine,
                vector_db=self.vector_db,
                ws_broadcaster=self.ws_broadcaster
            )
            worker.start()
            self.workers[camera_id] = worker
            return True

    def stop_stream(self, camera_id: str) -> bool:
        """Stops and terminates a stream monitoring worker."""
        with self._lock:
            worker = self.workers.get(camera_id)
            if not worker:
                return False
            worker.stop()
            del self.workers[camera_id]
            return True

    def list_active_streams(self) -> List[Dict[str, Any]]:
        """Returns details of all active camera monitoring workers."""
        with self._lock:
            active = []
            for cam_id, worker in self.workers.items():
                active.append({
                    "camera_id": cam_id,
                    "location": worker.location,
                    "stream_source": worker.stream_source,
                    "fps": round(worker.current_fps, 2),
                    "reconnect_count": worker.reconnect_count
                })
            return active
