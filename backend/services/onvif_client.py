"""Small, isolated adapter around the optional ONVIF client dependency."""

from __future__ import annotations

from urllib.parse import urlparse


class OnvifUnavailable(RuntimeError):
    pass


class OnvifConnectionError(RuntimeError):
    pass


def inspect_camera(endpoint: str, username: str, password: str) -> dict:
    """Authenticate, read device data, and resolve RTSP URIs dynamically.

    onvif-zeep is intentionally isolated here so discovery can still work on a
    server where the optional vendor-neutral ONVIF package was not installed.
    """
    try:
        from onvif import ONVIFCamera
    except ImportError as exc:
        raise OnvifUnavailable(
            "ONVIF support is not installed; install onvif-zeep in backend/.venv"
        ) from exc

    parsed = urlparse(endpoint)
    if not parsed.hostname:
        raise OnvifConnectionError("Invalid ONVIF endpoint")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        camera = ONVIFCamera(parsed.hostname, port, username, password)
        device_service = camera.create_devicemgmt_service()
        media_service = camera.create_media_service()
        info = device_service.GetDeviceInformation()
        profiles = media_service.GetProfiles()
        stream_profiles: list[dict] = []
        for index, profile in enumerate(profiles):
            uri = media_service.GetStreamUri(
                {
                    "StreamSetup": {
                        "Stream": "RTP-Unicast",
                        "Transport": {"Protocol": "RTSP"},
                    },
                    "ProfileToken": profile.token,
                }
            ).Uri
            encoder = getattr(profile, "VideoEncoderConfiguration", None)
            resolution = getattr(encoder, "Resolution", None)
            rate = getattr(encoder, "RateControl", None)
            stream_profiles.append(
                {
                    "profile_token": profile.token,
                    "stream_type": "main" if index == 0 else "sub",
                    "codec": str(getattr(encoder, "Encoding", "unknown")),
                    "width": getattr(resolution, "Width", None),
                    "height": getattr(resolution, "Height", None),
                    "fps": getattr(rate, "FrameRateLimit", None),
                    "rtsp_uri": uri,
                }
            )
        if not stream_profiles:
            raise OnvifConnectionError("Camera returned no usable ONVIF media profiles")
        return {
            "ip_address": parsed.hostname,
            "manufacturer": getattr(info, "Manufacturer", None),
            "model": getattr(info, "Model", None),
            "streams": stream_profiles,
        }
    except OnvifUnavailable:
        raise
    except Exception as exc:
        raise OnvifConnectionError(
            "ONVIF authentication or capability discovery failed"
        ) from exc
