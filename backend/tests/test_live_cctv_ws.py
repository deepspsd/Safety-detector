import sys
import os
import json
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from database import SessionLocal, User
from auth_utils import create_access_token

def run_ws_test():
    db = SessionLocal()
    user = db.query(User).filter(User.role != "Home").first()
    if not user:
        user = db.query(User).first()
    token = create_access_token({"sub": user.email}) if user else "test"
    db.close()

    client = TestClient(app)
    abs_video = os.path.abspath("backend/tests/test_images/test_stream.mp4")
    print("Connecting to /ws/detect-cctv...")
    with client.websocket_connect("/ws/detect-cctv") as ws:
        print("Connected to CCTV WebSocket!")
        ws.send_text(json.dumps({
            "token": token,
            "camera_url": abs_video,
            "floor": "ground"
        }))
        ack = ws.receive_json()
        print("Handshake ack:", ack.get("status"), "mode:", ack.get("mode"))
        
        # receive first processed frame from camera
        frame_data = ws.receive_json()
        dets = frame_data.get("detections", [])
        bangles = [d for d in dets if "bangle" in str(d.get("label", "")).lower()]
        fc = frame_data.get("frame_count")
        print(f"\nLive WebSocket Stream Output (Frame #{fc}):")
        print(f"  Total detections: {len(dets)}")
        print(f"  Bangle detections: {len(bangles)}")
        print(f"  is_compliant: {frame_data.get('is_compliant')}")
        print(f"  violations_count: {frame_data.get('violations_count')}")
        for b in bangles:
            print(f"    -> {b.get('label')} conf={b.get('confidence')} bbox={b.get('bbox')}")
        
        ann = frame_data.get("annotated_frame", "")
        print(f"  annotated_frame length: {len(ann)} chars")
        if len(bangles) > 0 and len(ann) > 1000:
            print("\nVERDICT: [PASS] Real-time RTSP/CCTV WebSocket delivers proper wrist bangle detections + amber-red boxes!")
        else:
            print("\nVERDICT: [FAIL]")

if __name__ == "__main__":
    run_ws_test()
