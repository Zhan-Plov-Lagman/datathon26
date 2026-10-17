from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path

from app.video.sampling import check_deadline, plan_windows, probe_video, sample_frames


class LocalVideoAnalyzer:
    """Reusable observation extractor. Keep the observer loaded across API jobs.

    Role B consumes the windows; role C must not map empty observations directly
    to a no_violation verdict without completed rule evaluation.
    """
    def __init__(self, observer, window_sec: float = 12, overlap_sec: float = 2,
                 frames_per_window: int = 6, max_long_edge: int = 560):
        self.observer = observer
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec
        self.frames_per_window = frames_per_window
        self.max_long_edge = max_long_edge

    def analyze(self, video_path: str | Path, job_id: str, timeout_sec: float = 240,
                start_sec: float = 0, end_sec: float | None = None) -> dict:
        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("Timeout must be finite and positive")
        started = time.monotonic()
        deadline = started + timeout_sec
        info = probe_video(video_path, deadline)
        windows = plan_windows(info.duration_sec, self.window_sec, self.overlap_sec,
                               self.frames_per_window, start_sec, end_sec)
        selected = sample_frames(video_path, [t for w in windows for t in w.targets], info,
                                 self.max_long_edge, deadline)
        decode_sec = time.monotonic() - started
        results = []
        previous_context = ""
        for number, window in enumerate(windows):
            check_deadline(deadline)
            frames = []
            seen = set()
            for target in window.targets:
                frame = selected[target]
                if frame.timestamp_sec not in seen:
                    seen.add(frame.timestamp_sec)
                    frames.append(frame)
            inference_start = time.monotonic()
            body = self.observer.observe(frames, previous_context, deadline)
            check_deadline(deadline)
            results.append({"window_id": number, "window_start_sec": window.start_sec,
                            "window_end_sec": window.end_sec,
                            "frame_timestamps": [f.timestamp_sec for f in frames], **body,
                            "inference_sec": round(time.monotonic() - inference_start, 3)})
            previous_context = body["operation_context"]
        check_deadline(deadline)
        return {"schema_version": "observations.v1", "job_id": job_id,
                "analysis_kind": "observations_only", "complete": True,
                "coverage_scope": "full_video" if start_sec == 0 and end_sec in (None, info.duration_sec) else "interval",
                "video": asdict(info), "windows": results,
                "model": {"id": getattr(self.observer, "model_id", "test-observer"),
                          "revision": getattr(self.observer, "model_revision", "test"),
                          "quantization": getattr(self.observer, "quantization", "none")},
                "settings": {"window_sec": self.window_sec, "overlap_sec": self.overlap_sec,
                             "frames_per_window": self.frames_per_window, "max_long_edge": self.max_long_edge},
                "metrics": {"total_sec": round(time.monotonic() - started, 3),
                            "decode_sec": round(decode_sec, 3),
                            **(self.observer.metrics() if hasattr(self.observer, "metrics") else {})}}
