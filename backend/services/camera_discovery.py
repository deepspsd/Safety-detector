"""Vendor-neutral ONVIF WS-Discovery with a constrained private-LAN fallback."""
from __future__ import annotations

import ipaddress
import socket
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Iterable
from urllib.parse import urlparse

WS_DISCOVERY_ADDRESS = ("239.255.255.250", 3702)
WSA = "http://www.w3.org/2005/08/addressing"
WSD = "http://docs.oasis-open.org/ws-dd/ns/discovery/2009/01"


@dataclass(frozen=True)
class DiscoveredDevice:
    device_id: str
    ip_address: str
    onvif_endpoint: str
    manufacturer: str | None = None
    model: str | None = None
    services: tuple[str, ...] = ()
    status: str = "discovered"

    def public(self) -> dict:
        data = asdict(self)
        data["services"] = list(self.services)
        return data


def _probe_xml() -> bytes:
    message_id = f"uuid:{uuid.uuid4()}"
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
 xmlns:w="{WSA}" xmlns:d="{WSD}">
 <e:Header><w:MessageID>{message_id}</w:MessageID><w:To>urn:docs-oasis-open-org:ws-dd-ns:discovery</w:To><w:Action>http://docs.oasis-open.org/ws-dd/ns/discovery/2009/01/Probe</w:Action></e:Header>
 <e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>
</e:Envelope>'''.replace("dn:", "").encode("utf-8")


def _parse_probe_match(payload: bytes) -> list[DiscoveredDevice]:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return []
    locations = root.findall(f".//{{{WSD}}}XAddrs")
    devices: list[DiscoveredDevice] = []
    for location in locations:
        for endpoint in (location.text or "").split():
            parsed = urlparse(endpoint)
            if parsed.hostname:
                devices.append(DiscoveredDevice(
                    device_id=str(uuid.uuid5(uuid.NAMESPACE_URL, endpoint)),
                    ip_address=parsed.hostname,
                    onvif_endpoint=endpoint,
                ))
    return devices


def discover_onvif(timeout_seconds: float = 5.0, interface: str | None = None, retries: int = 1) -> list[dict]:
    """Broadcast ONVIF WS-Discovery and return deduplicated device endpoints.

    `interface` may be a local IPv4 address on a multi-NIC server. This never
    scans the public Internet; it only multicasts on the selected local LAN.
    """
    timeout_seconds = max(1.0, min(float(timeout_seconds), 20.0))
    retries = max(1, min(int(retries), 3))
    found: dict[str, DiscoveredDevice] = {}
    for _ in range(retries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 0)
            if interface:
                sock.bind((interface, 0))
            sock.settimeout(0.25)
            sock.sendto(_probe_xml(), WS_DISCOVERY_ADDRESS)
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                try:
                    data, _ = sock.recvfrom(65_535)
                except socket.timeout:
                    continue
                for device in _parse_probe_match(data):
                    found[device.onvif_endpoint] = device
        finally:
            sock.close()
    return [device.public() for device in found.values()]


def allowed_private_subnet(subnet: str) -> ipaddress.IPv4Network:
    network = ipaddress.ip_network(subnet, strict=False)
    if not isinstance(network, ipaddress.IPv4Network) or not network.is_private:
        raise ValueError("Discovery fallback is restricted to an explicit RFC1918 IPv4 subnet")
    if network.num_addresses > 1024:
        raise ValueError("Discovery fallback subnet must contain at most 1024 addresses")
    return network


def probe_onvif_endpoints(subnet: str, ports: Iterable[int] = (80, 8080, 8899), timeout_seconds: float = 0.25) -> list[dict]:
    """Conservative local fallback. It only finds reachable HTTP endpoints;
    callers must authenticate and validate ONVIF before registration.
    """
    network = allowed_private_subnet(subnet)
    results: list[dict] = []
    for host in network.hosts():
        host_str = str(host)
        for port in ports:
            try:
                with socket.create_connection((host_str, int(port)), timeout=timeout_seconds):
                    results.append(DiscoveredDevice(
                        device_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"onvif-candidate:{host_str}:{port}")),
                        ip_address=host_str,
                        onvif_endpoint=f"http://{host_str}:{port}/onvif/device_service",
                        status="candidate_requires_validation",
                    ).public())
                    break
            except OSError:
                continue
    return results


# ─────────────────────────────────────────────────────────────────────────────
# HikVision / ONVIF Quick-Add helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_hikvision_rtsp_url(
    ip: str,
    username: str,
    password: str,
    channel: int = 1,
    stream: str = "sub",   # "main" (1080p+) | "sub" (D1/CIF, recommended for AI)
    port: int = 554,
) -> str:
    """
    Build the RTSP stream URL for a HikVision IP camera.

    Stream types
    ─────────────
      main → /Streaming/Channels/{channel}01  — full resolution, high bitrate
      sub  → /Streaming/Channels/{channel}02  — sub-stream, lower bitrate (AI-recommended)

    Example output:
      rtsp://admin:pass123@192.168.1.64:554/Streaming/Channels/102

    Usage in Settings UI (HikVision Quick-Add form):
      POST /cameras with rtsp_url = build_hikvision_rtsp_url(ip, user, pass, channel, stream)

    Note: Credentials are embedded in the URL only for RTSP transport.
    They are encrypted at rest in the camera_credentials table.
    """
    suffix = "01" if stream == "main" else "02"
    path   = f"/Streaming/Channels/{channel}{suffix}"
    # URL-encode credentials to handle special characters
    from urllib.parse import quote
    u = quote(username, safe="")
    p = quote(password, safe="")
    return f"rtsp://{u}:{p}@{ip}:{port}{path}"


def hikvision_quick_add(
    name: str,
    ip: str,
    username: str,
    password: str,
    floor: str = "ground",
    channel: int = 1,
    ai_stream: str = "sub",
    display_stream: str = "main",
) -> dict:
    """
    Return a dict ready to POST to /cameras for a HikVision camera.
    The frontend Quick-Add form calls this via POST /cameras/hikvision-quick-add.

    Returns
    -------
    dict with all fields needed by the /cameras POST endpoint.
    """
    return {
        "name":             name,
        "floor":            floor,
        "manufacturer":     "Hikvision",
        "rtsp_url":         build_hikvision_rtsp_url(ip, username, password, channel, ai_stream),
        "preferred_stream": display_stream,
        "ai_stream":        ai_stream,
        "ip_address":       ip,
        "status":           "online",
    }
