"""Quick RTSP connection test"""
import cv2
import os

# Force TCP transport for RTSP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

rtsp_url = "rtsp://admin:Password@2026@192.168.1.131:554/video/live?channel=1&subtype=0"
print(f"Testing connection to: {rtsp_url}")

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

if not cap.isOpened():
    print("❌ FAILED to open RTSP stream")
    print("Possible issues:")
    print("  1. Wrong credentials (admin:Password@2026)")
    print("  2. Wrong RTSP URL path")
    print("  3. Camera not responding")
    print("  4. Network issue (but TCP port 554 is reachable)")
else:
    print("✓ Stream opened successfully")
    ret, frame = cap.read()
    if ret and frame is not None:
        print(f"✓ Frame read successfully: {frame.shape}")
    else:
        print("❌ Failed to read frame from stream")

cap.release()
