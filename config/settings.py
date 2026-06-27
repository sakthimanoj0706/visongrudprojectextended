import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "database"
REGISTRY_DIR = BASE_DIR / "registry"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure all critical directories exist
for directory in [DB_DIR, REGISTRY_DIR, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database Config
DB_PATH = DB_DIR / "visionguard.db"

# Model Configuration
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# Model URLs
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# Face Detection Settings
DETECTION_THRESHOLD = 0.90
NMS_THRESHOLD = 0.30
TOP_K = 5000

# Face Recognition Settings
# OpenCV SFace Cosine Similarity Threshold:
# Note: SFace's official threshold for 0.1% False Acceptance Rate is 0.363.
# However, to align with the user's specification, we set a default of 0.60,
# which can be adjusted via command line or settings.
SIMILARITY_THRESHOLD = 0.60

# Video Processing Settings
FRAME_SKIP = 0          # Set to >0 to process every N-th frame (e.g. 5) for faster search
RESIZE_WIDTH = None     # Downscaling width for faster CPU detection (None means native)

# FAISS Vector Database Settings
VECTOR_INDEX_PATH = DB_DIR / "face_index.faiss"
EMBEDDING_DIMENSION = 128 # Configurable vector dimension (128 for SFace, 512 for ArcFace)

# JWT Security Settings
JWT_SECRET_KEY = "anti_gravity_secret_key"
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Real-Time Surveillance & Evidence Storage Settings
EVIDENCE_DIR = BASE_DIR / "evidence"
ALERT_COOLDOWN_SECONDS = 60
ESCALATION_TIME_WINDOW_MINUTES = 5
MAX_RECONNECT_ATTEMPTS = 5

# Alert Engine Settings (Phase 5)
ALERT_RETENTION_DAYS = 30
MOCK_EMAIL_SINK = "alerts@visionguard.local"
MOCK_SMS_SINK = "+15550199"

# Multi-Camera Person Tracking Settings (Phase 6)
REID_MODEL_PATH = MODELS_DIR / "person_reid_youtu_2021nov.onnx"
REID_URL = "https://huggingface.co/opencv/person_reid_youtureid/resolve/main/person_reid_youtu_2021nov.onnx"
REID_THRESHOLD = 0.65
TRACKLET_TIMEOUT = 5.0  # Default tracklet inactivity timeout in seconds
TRACKING_DETECTION_MODE = "face-guided"  # Options: "face-guided", "yolo"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.onnx"
YOLO_URL = "https://github.com/hpc203/yolov8-opencv-dnn-cpp-python/raw/main/yolov8n.onnx"

# RAG Surveillance Memory Settings (Phase 7)
RAG_INDEX_PATH = DB_DIR / "text_memory.faiss"
TEXT_EMBEDDING_MODE = "dstv"  # Options: "dstv", "sentence-transformers"
RAG_EMBEDDING_DIM = 384

# Assistant Settings (Phase 8)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)

# NVIDIA GPU Optimization Settings (Phase 9)
def is_cuda_supported() -> bool:
    try:
        import cv2
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

_use_cuda_env = os.environ.get("USE_CUDA", "AUTO").upper()
if _use_cuda_env == "TRUE":
    USE_CUDA = True
elif _use_cuda_env == "FALSE":
    USE_CUDA = False
else:
    USE_CUDA = is_cuda_supported()

CUDA_FP16 = os.environ.get("CUDA_FP16", "FALSE").lower() in ("true", "1", "yes")
CUDA_DEVICE_INDEX = int(os.environ.get("CUDA_DEVICE_INDEX", "0"))

def resolve_dnn_backend_target():
    import cv2
    if not USE_CUDA:
        return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU, "CPU"
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU, "CPU (No GPU device)"
        cv2.cuda.setDevice(CUDA_DEVICE_INDEX)
        backend = cv2.dnn.DNN_BACKEND_CUDA
        if CUDA_FP16:
            target = cv2.dnn.DNN_TARGET_CUDA_FP16
            desc = f"CUDA FP16 (Device {CUDA_DEVICE_INDEX})"
        else:
            target = cv2.dnn.DNN_TARGET_CUDA
            desc = f"CUDA (Device {CUDA_DEVICE_INDEX})"
        return backend, target, desc
    except Exception as e:
        print(f"[GPU WARNING] CUDA initialization failed: {e}. Falling back to CPU.")
        return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU, "CPU (Fallback)"

# Ensure evidence directory exists
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)




