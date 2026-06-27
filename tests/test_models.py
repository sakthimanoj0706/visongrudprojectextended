import sys
from pathlib import Path
import cv2

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

def test_models_exist():
    assert settings.YUNET_MODEL_PATH.exists(), f"YuNet model missing at {settings.YUNET_MODEL_PATH}"
    assert settings.SFACE_MODEL_PATH.exists(), f"SFace model missing at {settings.SFACE_MODEL_PATH}"

def test_load_yunet():
    # Attempt to load detector with a tiny dummy input size (100x100)
    try:
        detector = cv2.FaceDetectorYN.create(
            model=str(settings.YUNET_MODEL_PATH),
            config="",
            input_size=(100, 100)
        )
        assert detector is not None, "Failed to instantiate FaceDetectorYN"
    except Exception as e:
        assert False, f"Exception occurred while loading YuNet: {e}"

def test_load_sface():
    try:
        recognizer = cv2.FaceRecognizerSF.create(
            model=str(settings.SFACE_MODEL_PATH),
            config=""
        )
        assert recognizer is not None, "Failed to instantiate FaceRecognizerSF"
    except Exception as e:
        assert False, f"Exception occurred while loading SFace: {e}"
