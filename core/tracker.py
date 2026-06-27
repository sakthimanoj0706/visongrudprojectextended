import time
import uuid
from typing import Dict, Any, List, Optional, Tuple

class Tracklet:
    def __init__(self, tracklet_id: str, camera_id: str, box: List[int], person_id: Optional[str] = None):
        self.tracklet_id = tracklet_id
        self.camera_id = camera_id
        self.last_box = box  # [x, y, w, h]
        self.person_id = person_id  # Assigned watchlist person_id or unknown_XXXX
        self.start_time = time.time()
        self.last_seen = time.time()
        self.frames_tracked = 1
        self.reid_embeddings: List[List[float]] = []  # Extracted ReID embeddings for this tracklet
        self.has_logged_db = False

    def update(self, box: List[int]):
        self.last_box = box
        self.last_seen = time.time()
        self.frames_tracked += 1

    def add_reid_embedding(self, embedding: List[float]):
        self.reid_embeddings.append(embedding)

    def get_average_reid(self) -> Optional[List[float]]:
        if not self.reid_embeddings:
            return None
        import numpy as np
        arr = np.array(self.reid_embeddings)
        mean_vec = np.mean(arr, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        return mean_vec.tolist()

class SingleCameraTracker:
    def __init__(self, camera_id: str, timeout_seconds: float = 5.0, min_iou: float = 0.25):
        self.camera_id = camera_id
        self.timeout_seconds = timeout_seconds
        self.min_iou = min_iou
        self.active_tracklets: Dict[str, Tracklet] = {}

    def _calculate_iou(self, boxA: List[int], boxB: List[int]) -> float:
        """Computes intersection-over-union of two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        denom = float(boxAArea + boxBArea - interArea)
        if denom <= 0:
            return 0.0
        return interArea / denom

    def update(self, detections: List[Dict[str, Any]]) -> Tuple[List[Tuple[Tracklet, Dict[str, Any]]], List[Tracklet]]:
        """
        Updates trackers with new detections.
        Each detection is a dict with 'box' [x, y, w, h] and other features.
        
        Returns:
            Tuple:
                - List of Tuples of (Tracklet, matching detection dict)
                - List of expired Tracklets to be cleaned up
        """
        current_time = time.time()

        # 1. Collect expired tracklets
        expired = []
        for tid, tracklet in list(self.active_tracklets.items()):
            if current_time - tracklet.last_seen > self.timeout_seconds:
                expired.append(tracklet)
                del self.active_tracklets[tid]

        # 2. Match detections with active trackers using IOU
        active_tids = list(self.active_tracklets.keys())
        matches: Dict[int, str] = {}  # detection_idx -> tracklet_id

        if active_tids and detections:
            used_det_indices = set()
            used_tids = set()

            candidates = []
            for det_idx, det in enumerate(detections):
                for tid in active_tids:
                    iou = self._calculate_iou(det["box"], self.active_tracklets[tid].last_box)
                    if iou >= self.min_iou:
                        candidates.append((iou, det_idx, tid))

            # Greedily match with highest IoU first
            candidates.sort(key=lambda x: x[0], reverse=True)
            for iou, det_idx, tid in candidates:
                if det_idx not in used_det_indices and tid not in used_tids:
                    matches[det_idx] = tid
                    used_det_indices.add(det_idx)
                    used_tids.add(tid)

        # 3. Update active tracklets or instantiate new ones
        results = []
        for det_idx, det in enumerate(detections):
            box = det["box"]
            if det_idx in matches:
                tid = matches[det_idx]
                tracklet = self.active_tracklets[tid]
                tracklet.update(box)
                results.append((tracklet, det))
            else:
                # Create a new tracklet session
                tid = f"TRK_{uuid.uuid4().hex[:8].upper()}"
                # If detection already has a recognized identity, propagate it
                person_id = det.get("person_id")
                tracklet = Tracklet(tid, self.camera_id, box, person_id)
                self.active_tracklets[tid] = tracklet
                results.append((tracklet, det))

        return results, expired
