import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from config import settings

class FaceRecognizerSFace:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"SFace ONNX model not found at {model_path}. Please run download_weights.py first.")
        
        self.model_path = str(model_path)
        backend_id, target_id, desc = settings.resolve_dnn_backend_target()
        try:
            self._recognizer = cv2.FaceRecognizerSF.create(
                model=self.model_path,
                config="",
                backend_id=backend_id,
                target_id=target_id
            )
            print(f"[DIAGNOSTICS] Face Recognizer (SFace) initialized with backend: {desc}")
        except Exception as e:
            print(f"[GPU WARNING] Face Recognizer (SFace) CUDA initialization failed: {e}. Falling back to CPU.")
            self._recognizer = cv2.FaceRecognizerSF.create(
                model=self.model_path,
                config="",
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU
            )
            print(f"[DIAGNOSTICS] Face Recognizer (SFace) initialized with backend: CPU (Fallback)")


    def align_face(self, frame: np.ndarray, face_raw: np.ndarray) -> np.ndarray:
        """
        Aligns and crops the face from the original frame based on YuNet landmark detection.
        
        Args:
            frame (np.ndarray): Original image/frame.
            face_raw (np.ndarray): Raw 15-D detection vector from YuNet.
            
        Returns:
            np.ndarray: Aligned and cropped face image (typically 112x112).
        """
        return self._recognizer.alignCrop(frame, face_raw)

    def extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extracts the 128-dimensional embedding vector from an aligned face.
        
        Args:
            aligned_face (np.ndarray): Aligned 112x112 face image.
            
        Returns:
            np.ndarray: 128-dimensional feature embedding (row vector of shape 1x128).
        """
        embedding = self._recognizer.feature(aligned_face)
        # Ensure the embedding is L2 normalized
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
