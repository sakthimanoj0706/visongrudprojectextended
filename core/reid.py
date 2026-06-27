import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from config import settings

def estimate_body_box(face_box: List[int], frame_shape: Tuple[int, int]) -> List[int]:
    """
    Estimates the full-body bounding box from a face bounding box.
    Uses anthropometric proportions to estimate the body region.
    """
    fx, fy, fw, fh = face_box
    img_h, img_w = frame_shape[:2]

    # Anthropometric approximation:
    # - Body width is roughly 3.5 to 4 times face width, centered on face.
    # - Body height is roughly 7 to 8 times face height, extending downwards.
    # - Start y slightly above face to capture hair/head.
    bx = int(fx - fw * 1.5)
    by = int(fy - fh * 0.2)
    bw = int(fw * 4.0)
    bh = int(fh * 7.5)

    # Clamp coordinates to frame boundaries
    bx = max(0, bx)
    by = max(0, by)
    bw = max(1, min(img_w - bx, bw))
    bh = max(1, min(img_h - by, bh))

    return [bx, by, bw, bh]

class PersonReIDExtractor:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"ReID ONNX model not found at {model_path}. Please run download_weights.py first.")
        
        self.model_path = str(model_path)
        self.net = cv2.dnn.readNet(self.model_path)
        backend_id, target_id, desc = settings.resolve_dnn_backend_target()
        try:
            self.net.setPreferableBackend(backend_id)
            self.net.setPreferableTarget(target_id)
            # Dry run to verify it doesn't crash on forward pass
            dummy_blob = np.zeros((1, 3, 256, 128), dtype=np.float32)
            self.net.setInput(dummy_blob)
            self.net.forward()
            print(f"[DIAGNOSTICS] ReID Extractor initialized with backend: {desc}")
        except Exception as e:
            print(f"[GPU WARNING] ReID Extractor CUDA initialization failed: {e}. Falling back to CPU.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print(f"[DIAGNOSTICS] ReID Extractor initialized with backend: CPU (Fallback)")


    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Extracts a 768-D normalized ReID embedding from a person body crop.
        Input crop is resized to 128x256.
        """
        if crop is None or crop.size == 0:
            return np.zeros(768, dtype=np.float32)


        # Standard Youtu ReID input: 128x256, scale factor 1.0/255.0, mean subtraction (0,0,0), swapRB=True
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 255.0, (128, 256), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        features = self.net.forward()
        
        # Flatten and L2 Normalize the feature vector
        vector = features.flatten()
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Computes cosine similarity between two normalized ReID embeddings."""
        return float(np.dot(vec1, vec2))

class YoloPersonDetector:
    def __init__(self, model_path: Path, confidence_threshold: float = 0.40):
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO ONNX model not found at {model_path}. Please run download_weights.py first.")
        
        self.model_path = str(model_path)
        self.confidence_threshold = confidence_threshold
        self.net = cv2.dnn.readNet(self.model_path)
        backend_id, target_id, desc = settings.resolve_dnn_backend_target()
        try:
            self.net.setPreferableBackend(backend_id)
            self.net.setPreferableTarget(target_id)
            # Dry run to verify it doesn't crash on forward pass
            dummy_blob = np.zeros((1, 3, 640, 640), dtype=np.float32)
            self.net.setInput(dummy_blob)
            self.net.forward()
            print(f"[DIAGNOSTICS] YOLO Detector initialized with backend: {desc}")
        except Exception as e:
            print(f"[GPU WARNING] YOLO Detector CUDA initialization failed: {e}. Falling back to CPU.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print(f"[DIAGNOSTICS] YOLO Detector initialized with backend: CPU (Fallback)")


    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects person bodies in the frame.
        Returns a list of dicts with 'box' [x, y, w, h] and 'confidence'.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        # YOLOv8 input is 640x640
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (640, 640), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()

        # Post-process YOLOv8 output: (1, 84, 8400)
        outputs = np.squeeze(outputs)
        if outputs.ndim < 2:
            return []
            
        # Transpose output to (8400, 84)
        outputs = outputs.T

        boxes = []
        confidences = []

        for row in outputs:
            # Person class index is 4 in COCO dataset (first class)
            confidence = row[4]
            if confidence >= self.confidence_threshold:
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                
                # Convert back to native scale and top-left coordinates
                x = int((cx - bw / 2) * (w / 640.0))
                y = int((cy - bh / 2) * (h / 640.0))
                width = int(bw * (w / 640.0))
                height = int(bh * (h / 640.0))

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.45)
        
        results = []
        for i in indices:
            idx = i[0] if isinstance(i, (list, np.ndarray)) else i
            results.append({
                "box": boxes[idx],
                "confidence": confidences[idx]
            })
        return results
