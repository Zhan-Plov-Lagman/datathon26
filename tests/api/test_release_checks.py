"""Release tools use synthetic files and never print responses or secrets."""
import argparse
import json
from unittest.mock import Mock

import httpx
import pytest

from scripts import check_e2e, run_local


@pytest.mark.parametrize("url", ["http://example.com", "https://user:password@example.com",
                                "https://example.com/?key=secret", "https://example.com/path",
                                "https://example.com/#fragment"])
def test_smoke_rejects_unsafe_origins(url):
    with pytest.raises(argparse.ArgumentTypeError):
        check_e2e.api_url(url)


def test_real_api_smoke_path_and_redacted_metrics(settings, result, tmp_path):
    from tests.api.test_api import clients, claim
    public, private = clients(settings)
    video = tmp_path / "synthetic.mov"
    video.write_bytes(b"synthetic transport fixture")
    original = public.post
    seen = []

    def post(path, **kwargs):
        response = original(path, **kwargs)
        if path == "/api/v1/analyze":
            seen.append(response.json()["job_id"])
            lease = claim(private)
            private.post(f"/internal/jobs/{lease['job_id']}/complete", json={
                "worker_id": "gpu-a", "attempt_token": lease["attempt_token"], "result": result}).raise_for_status()
        return response

    with public, private:
        private.post("/internal/workers/heartbeat", json={
            "worker_id": "gpu-a", "ready": True, "pipeline_version": "synthetic"}).raise_for_status()
        public.post = post
        metrics = check_e2e.run_check(public, video, interval=0)
    assert metrics["completed"] and metrics["jpeg_frames_checked"] == 1
    assert "description" not in json.dumps(metrics)
    assert seen[0] not in json.dumps(metrics)


@pytest.mark.parametrize("mode,code", [("not_ready", "analysis_not_ready"),
    ("failed", "analysis_failed"), ("wrong_id", "job_id_mismatch"),
    ("candidate", "invalid_status_schema"), ("invalid_jpeg", "invalid_evidence_or_event")])
def test_smoke_failure_paths_never_accept_or_echo_scene(mode, code, tmp_path, result):
    video = tmp_path / "synthetic.mov"
    video.write_bytes(b"synthetic")

    def handler(request):
        if request.url.path == "/api/v1/readiness":
            if mode == "not_ready":
                return httpx.Response(503, json={"ready": False})
            return httpx.Response(200, json={"ready": True})
        if request.url.path == "/api/v1/analyze":
            return httpx.Response(202, json={"job_id": "test"})
        if mode == "invalid_jpeg":
            body = {"job_id": "test", "status": "done", "verdict": "violation",
                    "violations": result["violations"]}
            body["violations"][0]["evidence_frames"][0]["image_base64"] = "PRIVATE_SYNTHETIC"
        else:
            body = {"job_id": "other" if mode == "wrong_id" else "test",
                    "status": "candidate" if mode == "candidate" else "failed"}
        return httpx.Response(200, json=body)

    with httpx.Client(base_url="http://127.0.0.1", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(check_e2e.CheckFailed, match=code) as error:
            check_e2e.run_check(client, video, interval=0)
        assert "PRIVATE_SYNTHETIC" not in str(error.value)


def test_smoke_overall_deadline(tmp_path, monkeypatch):
    video = tmp_path / "synthetic.mov"
    video.write_bytes(b"synthetic")
    clock = [0.0]
    monkeypatch.setattr(check_e2e.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(check_e2e.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))

    def handler(request):
        if request.url.path.endswith("readiness"):
            return httpx.Response(200, json={"ready": True})
        if request.url.path.endswith("analyze"):
            return httpx.Response(202, json={"job_id": "test"})
        return httpx.Response(200, json={"job_id": "test", "status": "processing"})

    with httpx.Client(base_url="http://127.0.0.1", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(check_e2e.CheckFailed, match="deadline_exceeded"):
            check_e2e.run_check(client, video, timeout=3, interval=2)
    assert clock[0] == 3


def test_all_in_one_requires_real_evaluator_before_start(monkeypatch, capsys):
    monkeypatch.setattr(run_local, "load_dotenv", lambda **kwargs: None)
    monkeypatch.setattr(run_local.Settings, "from_env", lambda: None)
    monkeypatch.delenv("FORTE_RULES_FACTORY", raising=False)
    spawn = Mock()
    monkeypatch.setattr(run_local.subprocess, "Popen", spawn)
    assert run_local.main() == 2
    spawn.assert_not_called()
    assert "FORTE_RULES_FACTORY" in capsys.readouterr().out


def test_all_in_one_stops_coordinator_after_worker_exits(monkeypatch):
    monkeypatch.setattr(run_local, "load_dotenv", lambda **kwargs: None)
    monkeypatch.setattr(run_local.Settings, "from_env", lambda: None)
    monkeypatch.setenv("FORTE_RULES_FACTORY", "synthetic:configured_for_test_only")
    monkeypatch.setattr(run_local, "require_free_ports", lambda: None)
    monkeypatch.setattr(run_local, "wait_for_coordinator", lambda process: None)
    coordinator, worker = Mock(), Mock()
    coordinator.poll.return_value, worker.poll.return_value = None, 1
    spawn = Mock(side_effect=[coordinator, worker])
    monkeypatch.setattr(run_local.subprocess, "Popen", spawn)
    stop = Mock()
    monkeypatch.setattr(run_local, "stop_process", stop)
    assert run_local.main() == 1
    assert [call.args[0] for call in stop.call_args_list] == [worker, coordinator]
    worker_command = spawn.call_args_list[1].args[0]
    assert worker_command[-1] == "http://127.0.0.1:8001"
