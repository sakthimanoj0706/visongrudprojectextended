import argparse
import sys
import json
from pathlib import Path

from config import settings
from core.detector import FaceDetectorYuNet
from core.recognizer import FaceRecognizerSFace
from core.event_engine import EventEngine
from core.vector_db import VectorDBManager
from core.pipeline import SurveillancePipeline

def get_pipeline() -> SurveillancePipeline:
    """Instantiates the detector, recognizer, event engine, and vector db manager."""
    if not settings.YUNET_MODEL_PATH.exists() or not settings.SFACE_MODEL_PATH.exists():
        print("[ERROR] Model weights are missing. Please run the downloader first:")
        print("python models/download_weights.py")
        sys.exit(1)

    print("[INIT] Loading model weights into memory...")
    detector = FaceDetectorYuNet(
        model_path=settings.YUNET_MODEL_PATH,
        threshold=settings.DETECTION_THRESHOLD,
        nms_threshold=settings.NMS_THRESHOLD,
        top_k=settings.TOP_K
    )
    recognizer = FaceRecognizerSFace(model_path=settings.SFACE_MODEL_PATH)
    event_engine = EventEngine(db_path=settings.DB_PATH)
    
    # Initialize FAISS Vector database manager
    vector_db = VectorDBManager(
        index_path=settings.VECTOR_INDEX_PATH,
        dimension=settings.EMBEDDING_DIMENSION,
        event_engine=event_engine
    )
    
    return SurveillancePipeline(detector, recognizer, event_engine, vector_db)

def handle_enroll(args):
    pipeline = get_pipeline()
    image_path = Path(args.image)
    try:
        pipeline.enroll_target(
            person_id=args.id,
            name=args.name,
            category=args.category,
            risk_level=args.risk_level,
            image_path=image_path
        )
    except Exception as e:
        print(f"[ERROR] Enrollment failed: {e}", file=sys.stderr)
        sys.exit(1)

def handle_search(args):
    pipeline = get_pipeline()
    video_path = Path(args.video)
    
    output_name = f"annotated_all_{video_path.name}" if not args.target_id else f"annotated_{args.target_id}_{video_path.name}"
    output_video_path = Path(args.output_video) if args.output_video else settings.OUTPUTS_DIR / output_name
    
    # Override settings dynamically from CLI arguments
    if args.frame_skip is not None:
        settings.FRAME_SKIP = args.frame_skip
    if args.resize_width is not None:
        settings.RESIZE_WIDTH = args.resize_width

    try:
        results = pipeline.search_video(
            video_path=video_path,
            camera_id=args.camera_id,
            camera_location=args.camera_location,
            target_id=args.target_id,
            threshold=args.threshold,
            output_video_path=output_video_path
        )
        if args.target_id and results:
            timeline = results[0]
            print(f"\n[SUCCESS] Search completed. Found {timeline['total_sightings']} sightings of target '{timeline['person_name']}'.")
            print(f"Timeline reports generated in: {settings.REPORTS_DIR}")
        else:
            print("\n[SUCCESS] Watchlist search scan completed.")
            
        print(f"Annotated video saved to: {output_video_path}")
    except Exception as e:
        print(f"[ERROR] Video search failed: {e}", file=sys.stderr)
        sys.exit(1)

def handle_timeline(args):
    event_engine = EventEngine(db_path=settings.DB_PATH)
    pipeline = SurveillancePipeline(None, None, event_engine, None)
    
    try:
        timeline = pipeline.timeline_engine.generate_timeline(args.target_id)
        print(json.dumps(timeline, indent=4))
    except Exception as e:
        print(f"[ERROR] Timeline generation failed: {e}", file=sys.stderr)
        sys.exit(1)

def handle_server(args):
    import uvicorn
    print(f"[SERVER] Starting VisionGuard API on http://{args.host}:{args.port} ...")
    uvicorn.run("api.server:app", host=args.host, port=args.port, reload=args.reload)

def main():
    parser = argparse.ArgumentParser(
        description="VisionGuard Phase 2: Hybrid SQLite + FAISS Vector Face Embedding Database CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to execute")

    # Subcommand: enroll
    parser_enroll = subparsers.add_parser("enroll", help="Enroll a target person into the watchlist registry")
    parser_enroll.add_argument("--id", required=True, help="Unique identifier for the target (e.g. target_001)")
    parser_enroll.add_argument("--name", required=True, help="Full name of the target person")
    parser_enroll.add_argument("--category", default="Watchlist", help="Classification category (e.g. Watchlist, VIP)")
    parser_enroll.add_argument("--risk-level", default="High", choices=["Low", "Medium", "High", "Critical"],
                               help="Assigned security risk level (default: High)")
    parser_enroll.add_argument("--image", required=True, help="Path to the target's face image (JPEG/PNG)")

    # Subcommand: search
    parser_search = subparsers.add_parser("search", help="Scan a recorded video file to track watchlist targets")
    parser_search.add_argument("--video", required=True, help="Path to the recorded video file (MP4/AVI)")
    parser_search.add_argument("--camera-id", required=True, help="Identifier of the CCTV camera (e.g. CAM001)")
    parser_search.add_argument("--camera-location", required=True, help="Physical location description of the camera")
    parser_search.add_argument("--target-id", help="Optional target ID to filter results (if omitted, searches for all watchlisted persons)")
    parser_search.add_argument("--threshold", type=float, default=0.40, help="Cosine similarity threshold (default: 0.40)")
    parser_search.add_argument("--output-video", help="Optional custom path to save the annotated video")
    parser_search.add_argument("--frame-skip", type=int, help="Number of frames to skip during search (speedup factor)")
    parser_search.add_argument("--resize-width", type=int, help="Optional frame downscale width (e.g. 640) for CPU speedup")

    # Subcommand: timeline
    parser_timeline = subparsers.add_parser("timeline", help="Generate and view chronological sightings for a target")
    parser_timeline.add_argument("--target-id", required=True, help="ID of the target person")

    # Subcommand: server
    parser_server = subparsers.add_parser("server", help="Launch the FastAPI Watchlist Retrieval API Server")
    parser_server.add_argument("--host", default="127.0.0.1", help="API server bind address")
    parser_server.add_argument("--port", type=int, default=8000, help="API server port")
    parser_server.add_argument("--reload", action="store_true", help="Enable code hot-reloads")

    args = parser.parse_args()

    if args.command == "enroll":
        handle_enroll(args)
    elif args.command == "search":
        handle_search(args)
    elif args.command == "timeline":
        handle_timeline(args)
    elif args.command == "server":
        handle_server(args)

if __name__ == "__main__":
    main()
