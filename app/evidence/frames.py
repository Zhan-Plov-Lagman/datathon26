from __future__ import annotations

import base64
import io
from pathlib import Path

from app.video.sampling import probe_video, sample_frames


def extract_evidence(video_path: str | Path, timestamps: list[float],
                     deadline: float | None = None) -> list[dict]:
    """Return 1-3 source JPEG frames with actual PTS, at original dimensions."""
    if not 1 <= len(timestamps) <= 3:
        raise ValueError("Evidence requires 1-3 timestamps")
    info = probe_video(video_path, deadline)
    sampled = sample_frames(video_path, timestamps, info, deadline=deadline)
    result = []
    seen = set()
    for requested in timestamps:
        frame = sampled[requested]
        if frame.timestamp_sec in seen:
            continue
        seen.add(frame.timestamp_sec)
        output = io.BytesIO()
        frame.image.convert("RGB").save(output, format="JPEG", quality=95)
        result.append({"timestamp_sec": frame.timestamp_sec,
                       "image_base64": base64.b64encode(output.getvalue()).decode("ascii")})
    return result
