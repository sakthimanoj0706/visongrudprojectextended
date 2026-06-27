import cv2
import numpy as np
import json
import uuid
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from config import settings
from core.detector import FaceDetectorYuNet
from core.recognizer import FaceRecognizerSFace
from core.event_engine import EventEngine
from core.vector_db import VectorDBManager
from core.timeline import TimelineEngine

class SurveillancePipeline:
    def __init__(self, detector: FaceDetectorYuNet, recognizer: FaceRecognizerSFace,
                 event_engine: EventEngine, vector_db: VectorDBManager):
        self.detector = detector
        self.recognizer = recognizer
        self.event_engine = event_engine
        self.vector_db = vector_db
        self.timeline_engine = TimelineEngine(event_engine)

    def enroll_target(self, person_id: str, name: str, category: str, risk_level: str, image_path: Path) -> str:
        """
        Detects, aligns, and extracts an embedding from a target image,
        then registers the target details and embedding in the SQLite DB and FAISS index.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Target image not found at {image_path}")

        print(f"[ENROLL] Enrolling target person '{name}' (ID: {person_id}, Risk: {risk_level}) using {image_path.name}...")
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Detect face
        detections = self.detector.detect(img)
        if not detections:
            raise ValueError("No faces detected in the target image. Unable to enroll target.")
        
        # If multiple faces detected, use the largest one by bounding box area
        if len(detections) > 1:
            print(f"[WARNING] Multiple faces ({len(detections)}) detected. Using the largest face.")
            target_face = max(detections, key=lambda d: d["box"][2] * d["box"][3])
        else:
            target_face = detections[0]

        # Align and extract embedding
        aligned = self.recognizer.align_face(img, target_face["raw"])
        embedding = self.recognizer.extract_embedding(aligned)

        # Create unique directory for target in registry
        target_dir = settings.REGISTRY_DIR / person_id
        target_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filenames to allow multiple enrollments per person
        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        reg_image_path = target_dir / f"image_{timestamp_slug}.jpg"
        cv2.imwrite(str(reg_image_path), img)

        # 1. Register Person Details in SQLite (If not already present)
        self.event_engine.register_person(
            person_id=person_id,
            name=name,
            category=category,
            risk_level=risk_level
        )

        # 2. Add Face Embedding to Vector Database (FAISS index + face_embeddings SQL table)
        embedding_id = self.vector_db.add_face(
            person_id=person_id,
            embedding=embedding,
            image_path=str(reg_image_path),
            source_type="manual_enrollment",
            metadata={
                "enrollment_timestamp": datetime.utcnow().isoformat(),
                "crop_box": target_face["box"],
                "detection_confidence": target_face["confidence"]
            }
        )

        # Save FAISS index changes
        self.vector_db.save()
            
        print(f"[SUCCESS] Registered and indexed target embedding {embedding_id} for person '{name}' (ID: {person_id})")
        return embedding_id

    @staticmethod
    def format_timestamp(frame_number: int, fps: float) -> str:
        """Formats frame count and FPS into a CCTV HH:MM:SS timestamp string."""
        total_seconds = int(frame_number / fps) if fps > 0 else 0
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def search_video(self, video_path: Path, camera_id: str, camera_location: str,
                     target_id: Optional[str] = None, threshold: float = 0.40,
                     output_video_path: Optional[Path] = None,
                     progress_callback = None) -> List[Dict[str, Any]]:
        """
        Processes a video file, matches faces against the entire face embedding database (vector_db),
        logs events, captures crop screenshots, and outputs an annotated video.
        
        If target_id is specified, only events matching that target_id will be logged/drawn.
        Otherwise, ALL matching enrolled targets will be logged and highlighted.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found at {video_path}")

        if self.vector_db.ntotal == 0:
            raise ValueError("The vector database index is empty. Please enroll a target first.")

        # 1. Register Camera Metadata
        self.event_engine.register_camera(camera_id, camera_location, str(video_path))

        # 2. Initialize Video Reader
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 3. Initialize Video Writer if output path is requested
        writer = None
        if output_video_path:
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        print(f"[SEARCH] Processing video: {video_path.name} ({width}x{height} @ {fps:.2f} FPS)")
        if target_id:
            print(f"[SEARCH] Filtering for target_id: {target_id} (Threshold: {threshold})")
        else:
            print(f"[SEARCH] Scanning for all watchlist targets (Threshold: {threshold})")

        frame_count = 0
        matches_found = 0
        
        crops_dir = settings.OUTPUTS_DIR / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                
                # Frame skipping
                if settings.FRAME_SKIP > 0 and frame_count % (settings.FRAME_SKIP + 1) != 1:
                    if writer:
                        writer.write(frame)
                    continue

                # Run Face Detection
                det_frame = frame
                scale_x, scale_y = 1.0, 1.0
                if settings.RESIZE_WIDTH and width > settings.RESIZE_WIDTH:
                    new_w = settings.RESIZE_WIDTH
                    new_h = int(height * (new_w / width))
                    det_frame = cv2.resize(frame, (new_w, new_h))
                    scale_x = width / new_w
                    scale_y = height / new_h

                detections = self.detector.detect(det_frame)
                annotated_frame = frame.copy()

                for det in detections:
                    # Rescale bounding box to native resolution
                    box = det["box"]
                    native_box = [
                        int(box[0] * scale_x),
                        int(box[1] * scale_y),
                        int(box[2] * scale_x),
                        int(box[3] * scale_y)
                    ]
                    
                    # Align and extract embedding
                    aligned = self.recognizer.align_face(det_frame, det["raw"])
                    embedding = self.recognizer.extract_embedding(aligned)

                    # Query Vector Database (Top-1 nearest neighbor search)
                    matches = self.vector_db.search(
                        embedding=embedding,
                        k=1,
                        threshold=threshold,
                        query_source="video_search"
                    )

                    if matches:
                        # Extract the top match
                        match = matches[0]
                        matched_person_id = match["person_id"]
                        matched_name = match["name"]
                        similarity = match["similarity"]
                        risk_level = match["risk_level"]

                        # Apply filter if target_id is specified
                        if target_id and matched_person_id != target_id:
                            continue

                        matches_found += 1
                        timestamp_str = self.format_timestamp(frame_count, fps)
                        event_id = f"EVT_{uuid.uuid4().hex[:8].upper()}"

                        # Crop face crop
                        x, y, w, h_box = native_box
                        x, y = max(0, x), max(0, y)
                        w = min(width - x, w)
                        h_box = min(height - y, h_box)
                        
                        if w > 0 and h_box > 0:
                            crop = frame[y:y+h_box, x:x+w]
                            crop_path = crops_dir / f"{event_id}_{matched_person_id}.jpg"
                            cv2.imwrite(str(crop_path), crop)
                        
                        # Log Event in Database
                        match_details = {
                            "match": True,
                            "confidence": similarity,
                            "threshold": threshold,
                            "risk_level": risk_level,
                            "embedding_id": match["embedding_id"]
                        }
                        
                        self.event_engine.log_event(
                            event_id=event_id,
                            person_id=matched_person_id,
                            camera_id=camera_id,
                            video_source=video_path.name,
                            timestamp=timestamp_str,
                            frame_number=frame_count,
                            confidence=similarity,
                            bounding_box=native_box,
                            match_details=match_details
                        )

                        # Draw HUD annotations
                        # Color coding based on risk level
                        color = (0, 0, 255) # Red for critical/high
                        if risk_level in ["Low", "Medium"]:
                            color = (0, 255, 0) # Green for low/medium
                        elif risk_level == "High":
                            color = (0, 165, 255) # Orange for high

                        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h_box), color, 2)
                        
                        label = f"{matched_name} [{risk_level}] ({similarity:.2f})"
                        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(annotated_frame, (x, y - label_h - 10), (x + label_w, y), color, -1)
                        cv2.putText(annotated_frame, label, (x, y - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

                # Write frame to output video
                if writer:
                    writer.write(annotated_frame)

                if frame_count % 100 == 0:
                    percent_complete = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"[SEARCH] Processed {frame_count}/{total_frames} frames ({percent_complete:.1f}%) - Matches: {matches_found}")

                if progress_callback and (frame_count % 10 == 0 or frame_count == total_frames):
                    progress_callback(frame_count, total_frames, matches_found)

        finally:
            cap.release()
            if writer:
                writer.release()
            print(f"[SEARCH] Finished. Total Frames: {frame_count}. Total Matches: {matches_found}")

        # If a single target filter was applied, compile its specific timeline
        if target_id:
            timeline = self.timeline_engine.generate_timeline(target_id)
            json_report_path = settings.REPORTS_DIR / f"timeline_{target_id}.json"
            md_report_path = settings.REPORTS_DIR / f"timeline_{target_id}.md"
            self.timeline_engine.save_timeline_report(timeline, json_report_path, md_report_path)
            return [timeline]
            
        return []
