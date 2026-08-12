import asyncio
import logging
from typing import Dict, Set

from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

from services.camera_manager import get_latest_frame

log = logging.getLogger("webrtc_streamer")

class CameraStreamTrack(VideoStreamTrack):
    """
    A WebRTC VideoStreamTrack that yields the latest frame from the global camera_manager.
    This avoids making a second RTSP connection to the camera.
    """
    def __init__(self, camera_id: int):
        super().__init__()
        self.camera_id = camera_id
        # WebRTC needs frames at a consistent rate.
        # If the camera has no new frame, we send the last one.

    async def recv(self):
        # Determine the timestamp and wait for next frame time (e.g., 20 fps = 50ms)
        pts, time_base = await self.next_timestamp()

        # We don't want to poll tightly; sleep ~1/20th of a second
        await asyncio.sleep(0.05)

        # Get latest OpenCV BGR frame from CameraManager
        frame = get_latest_frame(self.camera_id)
        if frame is None:
            # If no frame yet, just sleep and raise exception or return a blank frame.
            # Returning a dummy 640x480 black frame to keep the connection alive.
            import numpy as np
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Convert OpenCV BGR to PyAV VideoFrame
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

class WebRTCManager:
    def __init__(self):
        # Keep track of active connections so they don't get garbage collected
        self._pcs: Set[RTCPeerConnection] = set()

    async def handle_offer(self, camera_id: int, offer_sdp: str, offer_type: str) -> dict:
        """
        Takes an SDP offer from the browser, creates a PeerConnection,
        adds our CameraStreamTrack, and returns the SDP answer.
        """
        pc = RTCPeerConnection()
        self._pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log.info(f"[WebRTC] Camera {camera_id} state is {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                self._pcs.discard(pc)

        # Add the video track linked to our camera manager
        video_track = CameraStreamTrack(camera_id=camera_id)
        pc.addTrack(video_track)

        # Set remote description (the offer from browser)
        offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
        await pc.setRemoteDescription(offer)

        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }

# Global singleton
webrtc_manager = WebRTCManager()
