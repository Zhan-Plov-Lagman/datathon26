import threading
import time
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.jobs.pipeline import IntegratedProcessor, create_processor
from app.jobs.worker import ProcessorProcess, execute_job
from tests.api.test_api import clients, upload, claim


def test_no_rules_means_no_production_processor(monkeypatch):
    monkeypatch.delenv("FORTE_RULES_FACTORY", raising=False)
    with pytest.raises(RuntimeError, match="FORTE_RULES_FACTORY"):
        create_processor()


def test_actual_transport_and_persistent_child(settings, tmp_path):
    public, private = clients(settings)
    processor = ProcessorProcess("tests.api.fakes:make_processor", startup_timeout=15)
    pid = processor.process.pid
    try:
        with public, private:
            for _ in range(2):
                job = upload(public).json()["job_id"]
                started = time.monotonic()
                lease = claim(private)
                assert execute_job(private, processor, lease, "gpu-a", tmp_path, started)
                assert public.get(f"/api/v1/jobs/{job}").json()["verdict"] == "no_violation"
                assert processor.process.pid == pid
            assert not list(tmp_path.glob("job-*"))
    finally:
        processor.close()


def test_stuck_gpu_process_is_terminated(tmp_path):
    processor = ProcessorProcess("tests.api.fakes:make_hanging_processor", startup_timeout=15)
    try:
        with pytest.raises(TimeoutError):
            processor.run({"video": "synthetic.mov", "output": str(tmp_path / "output.json"),
                           "job_id": "test", "timeout_sec": 0.2}, time.monotonic() + 0.2, threading.Event())
        assert not processor.process.is_alive()
    finally:
        processor.close()


def test_lease_heartbeat_during_work(settings, tmp_path):
    public, private = clients(settings)
    class SlowProcessor:
        def run(self, task, deadline, lost):
            time.sleep(0.25)
            return False
    with public, private:
        job = upload(public).json()["job_id"]
        lease = claim(private)
        lease["lease_sec"] = 0.15  # Test transport heartbeat interval, not DB lease length.
        calls = []
        original = private.post
        def post(url, **kwargs):
            if url.endswith("/heartbeat"):
                calls.append(url)
            return original(url, **kwargs)
        private.post = post
        assert not execute_job(private, SlowProcessor(), lease, "gpu-a", tmp_path, time.monotonic())
        assert calls
        assert public.get(f"/api/v1/jobs/{job}").json()["status"] == "failed"


def test_adapter_requires_full_coverage_and_reviewed_decision(result):
    report = {"schema_version": "observations.v1", "complete": True, "coverage_scope": "full_video",
              "video": {"duration_sec": 3.0}, "model": {"id": "synthetic", "revision": "test"}}
    from app.inference.worker import LocalGPUWorker
    from unittest.mock import Mock
    worker = LocalGPUWorker(object())
    worker.analyzer.analyze = Mock(return_value=report)
    event = result["violations"][0]
    decision = {"complete": True, "rules_reviewed": True, "rules_version": "test", "supported": [{
        "rule_id": "synthetic-rule", "type": event["type"], "anchor_sec": event["timecode_sec"],
        "description": event["description"], "evidence_timestamps": [1.0]}]}
    rules = SimpleNamespace(reviewed=True, evaluate=lambda *args, **kwargs: decision)
    worker.evidence = Mock(return_value={"0": event["evidence_frames"]})
    processor = IntegratedProcessor(worker, rules)
    assert processor.process("synthetic.mov", "test", 10)["verdict"] == "violation"
    assert worker.evidence.call_args.args[1] == {"0": [1.0]}
    assert worker.evidence.call_args.kwargs["deadline"] == worker.analyzer.analyze.call_args.kwargs["deadline"]
    report["coverage_scope"] = "interval"
    with pytest.raises(ValueError, match="Incomplete"):
        processor.process("synthetic.mov", "test", 10)
    report["coverage_scope"] = "full_video"
    decision["rules_reviewed"] = False
    with pytest.raises(ValidationError):
        processor.process("synthetic.mov", "test", 10)
    with pytest.raises(RuntimeError):
        IntegratedProcessor(worker, SimpleNamespace(reviewed=False))
