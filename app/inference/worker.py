"""In-process integration for C. Transport, queue and hard process timeout belong to C."""
from __future__ import annotations

from threading import Lock

from app.evidence.frames import extract_evidence_batch
from app.inference.pipeline import LocalVideoAnalyzer


class WorkerBusy(RuntimeError):
    """Caller must keep the job queued; do not run a second GPU task."""


class LocalGPUWorker:
    def __init__(self, observer):
        self.analyzer = LocalVideoAnalyzer(observer)
        self._lock = Lock()

    def _run(self, operation, *args, **kwargs):
        if not self._lock.acquire(blocking=False):
            raise WorkerBusy("GPU worker already has an active operation")
        try:
            return operation(*args, **kwargs)
        finally:
            self._lock.release()

    def analyze(self, video_path, job_id, timeout_sec=240, *, deadline=None):
        return self._run(self.analyzer.analyze, video_path, job_id, timeout_sec, deadline=deadline)

    def refine(self, video_path, job_id, intervals, timeout_sec=120, *, deadline=None):
        return self._run(self.analyzer.refine_intervals, video_path, job_id, intervals,
                         timeout_sec, deadline=deadline)

    def evidence(self, video_path, requests, *, deadline=None):
        return self._run(extract_evidence_batch, video_path, requests, deadline=deadline)

    def assess(self, video_path, report, timeout_sec=120, *, deadline=None, local_context=None):
        from app.inference.assessments import extract_assessments
        return self._run(extract_assessments, self.analyzer.observer, video_path, report,
                         timeout_sec, deadline=deadline, local_context=local_context)
