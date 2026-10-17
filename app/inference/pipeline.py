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
                start_sec: float = 0, end_sec: float | None = None, *,
                deadline: float | None = None) -> dict:
        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("Timeout must be finite and positive")
        started = time.monotonic()
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("Deadline must be finite")
        deadline = min(started + timeout_sec, deadline) if deadline is not None else started + timeout_sec
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

    def refine_intervals(self, video_path: str | Path, job_id: str, intervals: list[dict],
                         timeout_sec: float = 120, *, deadline: float | None = None) -> dict:
        """Denser review of B-selected intervals; never confirms a legal candidate.

        Each input has candidate_id, start_sec, end_sec. Identical intervals reuse
        one analysis. All intervals share one deadline; no partial-success return.
        """
        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("Timeout must be finite and positive")
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("Deadline must be finite")
        started = time.monotonic()
        deadline = min(started + timeout_sec, deadline) if deadline is not None else started + timeout_sec
        info = probe_video(video_path, deadline)
        normalized, identifiers = [], set()
        for interval in intervals:
            identifier = interval["candidate_id"]
            if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
                raise ValueError("Candidate identifiers must be nonempty and unique")
            start, end = interval["start_sec"], interval["end_sec"]
            if type(start) not in (int, float) or type(end) not in (int, float):
                raise ValueError("Interval endpoints must be numbers")
            plan_windows(info.duration_sec, start_sec=start, end_sec=end)
            normalized.append((identifier, start, end))
            identifiers.add(identifier)
        dense = LocalVideoAnalyzer(self.observer, window_sec=4, overlap_sec=1,
                                   frames_per_window=6, max_long_edge=self.max_long_edge)
        cache, results = {}, []
        for identifier, start, end in normalized:
            check_deadline(deadline)
            key = (start, end)
            if key not in cache:
                cache[key] = dense.analyze(video_path, job_id, timeout_sec,
                                           start_sec=start, end_sec=end, deadline=deadline)
            results.append({"candidate_id": identifier, "report": cache[key]})
        check_deadline(deadline)
        return {"schema_version": "refinements.v1", "job_id": job_id,
                "analysis_kind": "observations_only", "complete": True,
                "refinements": results, "metrics": {"unique_intervals": len(cache),
                "total_sec": round(time.monotonic() - started, 3)}}
