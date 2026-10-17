from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


class VideoError(RuntimeError):
    """An input cannot be decoded reliably; never a negative verdict."""


class DeadlineExceeded(TimeoutError):
    pass


def check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise DeadlineExceeded("Analysis deadline exceeded")


@dataclass(frozen=True)
class VideoInfo:
    duration_sec: float
    start_time_sec: float
    width: int
    height: int
    rotation: int


@dataclass(frozen=True)
class Window:
    start_sec: float
    end_sec: float
    targets: tuple[float, ...]


@dataclass
class SampledFrame:
    timestamp_sec: float
    image: object  # PIL.Image.Image; keep imports lazy for lightweight callers.


def probe_video(path: str | Path, deadline: float | None = None) -> VideoInfo:
    check_deadline(deadline)
    path = Path(path).resolve(strict=True)
    timeout = 20.0 if deadline is None else min(20.0, max(0.01, deadline - time.monotonic()))
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,start_time,duration:stream_side_data=rotation:format=duration",
             "-of", "json", str(path)],
            capture_output=True, timeout=timeout, check=True,
        )
        metadata = json.loads(result.stdout)
        stream = metadata["streams"][0]
        raw_duration = stream.get("duration", metadata["format"].get("duration"))
        if raw_duration in (None, "N/A"):
            raw_duration = metadata["format"].get("duration")
        duration = float(raw_duration)
        start_time = float(stream.get("start_time", 0) or 0)
        rotation = int(round(float(next(
            (entry["rotation"] for entry in stream.get("side_data_list", []) if "rotation" in entry), 0
        ))))
        if not math.isfinite(duration) or duration <= 0 or not math.isfinite(start_time):
            raise ValueError("Invalid timeline")
        if rotation % 90 != 0:
            raise ValueError("Only right-angle display rotation is supported")
        width, height = int(stream["width"]), int(stream["height"])
        if width <= 0 or height <= 0:
            raise ValueError("Invalid frame dimensions")
        return VideoInfo(duration, start_time, width, height, rotation % 360)
    except subprocess.TimeoutExpired as exc:
        raise DeadlineExceeded("Video probe timed out") from exc
    except (subprocess.CalledProcessError, FileNotFoundError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise VideoError("Unable to read video metadata; ffprobe and a valid video are required") from exc


def plan_windows(duration_sec: float, window_sec: float = 12, overlap_sec: float = 2,
                 frames_per_window: int = 6, start_sec: float = 0, end_sec: float | None = None) -> list[Window]:
    end = duration_sec if end_sec is None else end_sec
    if not all(math.isfinite(v) for v in (duration_sec, window_sec, overlap_sec, start_sec, end)):
        raise ValueError("Timeline values must be finite")
    if not (0 <= start_sec < end <= duration_sec):
        raise ValueError("Interval must be inside the video")
    if window_sec <= 0 or not 0 <= overlap_sec < window_sec or frames_per_window < 2:
        raise ValueError("Invalid window configuration")
    result: list[Window] = []
    start = start_sec
    while start < end:
        stop = min(start + window_sec, end)
        targets = tuple(round(start + (stop - start) * i / (frames_per_window - 1), 6)
                        for i in range(frames_per_window))
        result.append(Window(start, stop, targets))
        if stop >= end:
            break
        start += window_sec - overlap_sec
    return result


def sample_frames(path: str | Path, targets: list[float], info: VideoInfo,
                  max_long_edge: int | None = None, deadline: float | None = None) -> dict[float, SampledFrame]:
    """Decode once and select nearest real PTS for each requested time.

    No nominal FPS division and no generated/estimated source timestamps.
    Returned keys are requested times; values carry actual decoded times.
    """
    import av
    from PIL import Image

    requested = sorted(set(targets))
    if not requested:
        return {}
    if any(not math.isfinite(t) or t < 0 or t > info.duration_sec + 1e-6 for t in requested):
        raise ValueError("Requested frame is outside the video")
    if max_long_edge is not None and max_long_edge < 32:
        raise ValueError("Frame size must be at least 32 pixels")
    chosen: dict[float, SampledFrame] = {}
    cached: dict[float, SampledFrame] = {}
    index = 0
    previous = None
    previous_time = None

    def capture(frame, timestamp: float) -> SampledFrame:
        key = round(timestamp, 6)
        if key not in cached:
            image = frame.to_image()
            if info.rotation:
                image = image.rotate(info.rotation, expand=True)
            if max_long_edge:
                image.thumbnail((max_long_edge, max_long_edge), Image.Resampling.LANCZOS)
            cached[key] = SampledFrame(key, image)
        return cached[key]

    try:
        with av.open(str(Path(path).resolve(strict=True))) as container:
            if not container.streams.video:
                raise VideoError("No video stream")
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                check_deadline(deadline)
                if frame.pts is None or frame.time_base is None:
                    raise VideoError("Frame has no presentation timestamp")
                timestamp = float(frame.pts * frame.time_base) - info.start_time_sec
                if timestamp < -1e-4:
                    continue
                timestamp = max(0.0, timestamp)
                if previous_time is not None and timestamp < previous_time - 1e-6:
                    raise VideoError("Non-monotonic presentation timestamps")
                while index < len(requested) and requested[index] <= timestamp:
                    target = requested[index]
                    if previous is not None and abs(previous_time - target) <= abs(timestamp - target):
                        chosen[target] = capture(previous, previous_time)
                    else:
                        chosen[target] = capture(frame, timestamp)
                    index += 1
                previous, previous_time = frame, timestamp
                if index == len(requested):
                    break
            if previous is None:
                raise VideoError("No decodable frames")
            while index < len(requested):
                # Container duration includes the last frame's display interval.
                # A large gap means truncated decoding, not valid coverage.
                if requested[index] - previous_time > 0.5:
                    raise VideoError("Video ends before the requested interval")
                chosen[requested[index]] = capture(previous, previous_time)
                index += 1
    except (VideoError, DeadlineExceeded):
        raise
    except Exception as exc:
        raise VideoError("Video decoding failed") from exc
    return chosen
