import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from config import settings

class FaceDetectorYuNet:
    def __init__(self, model_path: Path, threshold: float = 0.90, nms_threshold: float = 0.30, top_k: int = 5000):
        if not model_path.exists():
            raise FileNotFoundError(f"YuNet ONNX model not found at {model_path}. Please run download_weights.py first.")
        
        self.model_path = str(model_path)
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._detector = None
        self._current_input_size = (0, 0) # (width, height)

    def _init_detector(self, size: Tuple[int, int]):
        """Initializes cv2.FaceDetectorYN with the specified size."""
        backend_id, target_id, desc = settings.resolve_dnn_backend_target()
        try:
            self._detector = cv2.FaceDetectorYN.create(
                model=self.model_path,
                config="",
                input_size=size,
                score_threshold=self.threshold,
                nms_threshold=self.nms_threshold,
                top_k=self.top_k,
                backend_id=backend_id,
                target_id=target_id
            )
            print(f"[DIAGNOSTICS] Face Detector (YuNet) initialized with backend: {desc}")
        except Exception as e:
            print(f"[GPU WARNING] Face Detector (YuNet) CUDA initialization failed: {e}. Falling back to CPU.")
            self._detector = cv2.FaceDetectorYN.create(
                model=self.model_path,
                config="",
                input_size=size,
                score_threshold=self.threshold,
                nms_threshold=self.nms_threshold,
                top_k=self.top_k,
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU
            )
            print(f"[DIAGNOSTICS] Face Detector (YuNet) initialized with backend: CPU (Fallback)")
        self._current_input_size = size


    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in the given frame.
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing:
                - 'box': [x, y, w, h] (bounding box coordinates)
                - 'landmarks': numpy array of shape (5, 2)
                - 'confidence': float (detection confidence)
                - 'raw': raw 15-D detection array from OpenCV
        """
        if frame is None:
            return []

        h, w = frame.shape[:2]
        size = (w, h)

        # Lazy initialization or dynamic resizing of the detector input size
        if self._detector is None:
            self._init_detector(size)
        elif self._current_input_size != size:
            self._detector.setInputSize(size)
            self._current_input_size = size

        retval, faces = self._detector.detect(frame)
        
        results = []
        if faces is not None:
            for face in faces:
                # Bounding box is at index 0-3 (x, y, width, height)
                # Cast coordinates to integers to prevent serialisation issues
                box = [int(x) for x in face[0:4]]
                # Landmarks are at index 4-13 (5 points x, y)
                landmarks = face[4:14].reshape(5, 2)
                # Confidence score is at index 14
                confidence = float(face[14])
                
                results.append({
                    "box": box,
                    "landmarks": landmarks,
                    "confidence": confidence,
                    "raw": face
                })
        return results
