import sys
from pathlib import Path
import requests

# Add root folder to path to import settings
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

def download_file(url: str, dest_path: Path):
    if dest_path.exists():
        print(f"[INFO] File already exists: {dest_path.name}")
        return

    print(f"[INFO] Downloading {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1 MB
    downloaded = 0
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rDownloading: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="")
                else:
                    print(f"\rDownloading: {downloaded} bytes", end="")
    print(f"\n[SUCCESS] Saved to {dest_path}")

def main():
    print("[START] Downloading model weights for VisionGuard...")
    try:
        download_file(settings.YUNET_URL, settings.YUNET_MODEL_PATH)
        download_file(settings.SFACE_URL, settings.SFACE_MODEL_PATH)
        download_file(settings.REID_URL, settings.REID_MODEL_PATH)
        if settings.TRACKING_DETECTION_MODE == "yolo":
            download_file(settings.YOLO_URL, settings.YOLO_MODEL_PATH)
        print("[SUCCESS] All model weights downloaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to download model weights: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

