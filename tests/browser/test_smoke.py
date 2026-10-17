"""Actual HTTP smoke tool through the worker transport; synthetic MOV and processor only."""
import threading
import time

import httpx

from app.api.server import create_public_app, create_worker_app
from app.jobs.store import JobStore
from app.jobs.worker import ProcessorProcess, execute_job
from scripts.check_e2e import run_check
from tests.api.conftest import settings  # noqa: F401
from tests.browser.test_ui import server, synthetic_mov  # noqa: F401


def test_http_smoke_with_isolated_synthetic_worker(settings, synthetic_mov, tmp_path):
    store = JobStore(settings)
    processor = ProcessorProcess("tests.api.fakes:make_processor", startup_timeout=15)
    stop = threading.Event()
    outcome = []
    try:
        with server(create_public_app(settings, store)) as public_url, server(create_worker_app(settings, store)) as private_url:
            with httpx.Client(base_url=private_url, headers={"X-Worker-Key": settings.worker_key},
                              trust_env=False, timeout=5) as private:
                hello = {"worker_id": "synthetic-test", "ready": True, "pipeline_version": "synthetic-test-only"}
                private.post("/internal/workers/heartbeat", json=hello).raise_for_status()

                def work_once():
                    try:
                        while not stop.is_set():
                            claimed_at = time.monotonic()
                            response = private.post("/internal/jobs/claim", json=hello)
                            response.raise_for_status()
                            if response.status_code == 204:
                                stop.wait(.02)
                                continue
                            outcome.append(execute_job(private, processor, response.json(), hello["worker_id"],
                                                       tmp_path, claimed_at))
                            return
                    except Exception:
                        outcome.append(False)

                thread = threading.Thread(target=work_once)
                thread.start()
                try:
                    with httpx.Client(base_url=public_url, headers={"X-API-Key": settings.api_key},
                                      trust_env=False, follow_redirects=False) as public:
                        metrics = run_check(public, synthetic_mov, timeout=15, interval=.02)
                        assert metrics["completed"]
                        assert metrics["events_count"] == metrics["jpeg_frames_checked"] == 0
                        assert metrics["elapsed_sec"] < 15
                finally:
                    stop.set()
                    thread.join(timeout=10)
                assert outcome == [True]
                assert not thread.is_alive()
    finally:
        stop.set()
        processor.close()
