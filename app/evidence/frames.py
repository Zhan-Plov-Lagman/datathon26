from __future__ import annotations

import base64
import io
from pathlib import Path

from app.video.sampling import check_deadline, probe_video, sample_frames


def extract_evidence(video_path: str | Path, timestamps: list[float],
                     deadline: float | None = None) -> list[dict]:
    """Return 1-3 source JPEG frames with actual PTS, at original dimensions."""
    return extract_evidence_batch(video_path, {"event": timestamps}, deadline)["event"]


def extract_evidence_batch(video_path: str | Path, requests: dict[str, list[float]],
                           deadline: float | None = None) -> dict[str, list[dict]]:
    """Decode once for all events, preserving each event's 1-3 actual JPEGs."""
    check_deadline(deadline)
    for identifier, timestamps in requests.items():
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Evidence identifiers must be nonempty strings")
        if not 1 <= len(timestamps) <= 3:
            raise ValueError("Evidence requires 1-3 timestamps per event")
    if not requests:
        return {}
    info = probe_video(video_path, deadline)
    sampled = sample_frames(video_path, [t for times in requests.values() for t in times], info, deadline=deadline)
    encoded = {}
    for frame in sampled.values():
        check_deadline(deadline)
        if frame.timestamp_sec not in encoded:
            output = io.BytesIO()
            frame.image.convert("RGB").save(output, format="JPEG", quality=95)
            encoded[frame.timestamp_sec] = {"timestamp_sec": frame.timestamp_sec,
                "image_base64": base64.b64encode(output.getvalue()).decode("ascii")}
    result = {}
    for identifier, timestamps in requests.items():
        seen, frames = set(), []
        for requested in timestamps:
            frame = sampled[requested]
            if frame.timestamp_sec not in seen:
                frames.append(dict(encoded[frame.timestamp_sec]))
                seen.add(frame.timestamp_sec)
        result[identifier] = frames
    check_deadline(deadline)
    return result
