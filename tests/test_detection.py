import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings
from core.detector import FaceDetectorYuNet

def test_detector_empty_frame():
    detector = FaceDetectorYuNet(
        model_path=settings.YUNET_MODEL_PATH,
        threshold=0.9,
        nms_threshold=0.3
    )
    
    # Create a blank black image
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    results = detector.detect(dummy_img)
    assert isinstance(results, list)
    # On a black image, detection should be empty
    assert len(results) == 0
