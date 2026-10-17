"""Local browser checks with synthetic MOV only. No external URLs or bank content."""
import contextlib
import os
import socket
import threading
import time
from pathlib import Path

import av
import httpx
import pytest
import uvicorn
from PIL import Image
from playwright.sync_api import sync_playwright, expect

from app.api.server import create_public_app, create_worker_app
from app.jobs.store import JobStore
from tests.api.conftest import result, settings  # noqa: F401 - shared synthetic fixtures


@contextlib.contextmanager
def server(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    instance = uvicorn.Server(uvicorn.Config(app, log_level="critical", access_log=False))
    thread = threading.Thread(target=instance.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not instance.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert instance.started
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        instance.should_exit = True
        thread.join(timeout=10)
        sock.close()


@pytest.fixture
def synthetic_mov(tmp_path):
    path = tmp_path / "synthetic.mov"
    with av.open(str(path), "w", format="mov") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        for index in range(30):
            frame = av.VideoFrame.from_image(Image.new("RGB", (96, 64), (0, 80 + index, 50)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_browser_upload_result_seek_resume_and_mobile(settings, result, synthetic_mov):
    store = JobStore(settings)
    with server(create_public_app(settings, store)) as url, server(create_worker_app(settings, store)) as private_url:
        with sync_playwright() as playwright:
            channel = os.environ.get("FORTE_BROWSER_CHANNEL", "msedge")
            browser = playwright.chromium.launch(channel=channel, headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors, external = [], []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("request", lambda request: external.append(request.url) if not request.url.startswith((url, "blob:", "data:")) else None)
            try:
                page.goto(url)
                page.locator("#api-key").fill("wrong")
                page.locator("#check-ready").click()
                expect(page.locator("#readiness")).to_contain_text("Ключ доступа не принят")
                page.locator("#api-key").fill(settings.api_key)
                page.locator("#check-ready").click()
                expect(page.locator("#readiness")).to_contain_text("Анализ пока недоступен")
                with httpx.Client(base_url=private_url, headers={"X-Worker-Key": settings.worker_key}, trust_env=False) as worker:
                    hello = {"worker_id": "browser-test", "ready": True, "pipeline_version": "synthetic-test-only"}
                    assert worker.post("/internal/workers/heartbeat", json=hello).status_code == 200
                    page.locator("#check-ready").click()
                    expect(page.locator("#readiness")).to_contain_text("Анализ доступен")
                    page.locator("#video-file").set_input_files(str(synthetic_mov))
                    page.locator("#submit").click()
                    expect(page.locator("#job-label")).to_contain_text("ID:")
                    job_id = page.locator("#job-id").input_value()
                    lease = worker.post("/internal/jobs/claim", json=hello).json()
                    assert lease["job_id"] == job_id
                    result["violations"][0]["description"] = "Synthetic <script>window.xss=true</script>"
                    assert worker.post(f"/internal/jobs/{job_id}/complete", json={
                        "worker_id": "browser-test", "attempt_token": lease["attempt_token"], "result": result}).status_code == 200
                    expect(page.locator("#status-badge")).to_have_text("Завершено", timeout=10000)
                    expect(page.locator(".event")).to_have_count(1)
                    assert page.locator(".frames img").evaluate("image => image.complete && image.naturalWidth === 16")
                    assert page.evaluate("window.xss") is None
                    page.wait_for_function("document.querySelector('video').readyState >= 1")
                    page.locator(".event-heading .time-button").click()
                    assert abs(page.locator("video").evaluate("video => video.currentTime") - 1.0) < .1
                    assert settings.api_key not in page.evaluate("JSON.stringify(localStorage)")
                    snapshots = Path(".cache/browser")
                    snapshots.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(snapshots / "desktop.png"), full_page=True)
                    page.set_viewport_size({"width": 390, "height": 844})
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                    page.screenshot(path=str(snapshots / "mobile.png"), full_page=True)
                    page.reload()
                    expect(page.locator("#api-key")).to_have_value("")
                    expect(page.locator("#job-id")).to_have_value(job_id)
                    page.locator("#api-key").fill(settings.api_key)
                    page.locator("summary").click()
                    page.locator("#resume-form button").click()
                    expect(page.locator("#status-badge")).to_have_text("Завершено")
                    expect(page.locator("#playback-note")).to_be_visible()
                    # A terminal failure must never render the successful negative verdict.
                    store.create("browser-failed")
                    failed_lease = worker.post("/internal/jobs/claim", json=hello).json()
                    assert worker.post("/internal/jobs/browser-failed/fail", json={
                        "worker_id": "browser-test", "attempt_token": failed_lease["attempt_token"],
                        "code": "invalid_video"}).status_code == 200
                    page.locator("#job-id").fill("browser-failed")
                    page.locator("#resume-form button").click()
                    expect(page.locator("#status-badge")).to_have_text("Ошибка")
                    expect(page.locator("#status-text")).to_contain_text("Результат о наличии нарушений не получен")
                assert errors == []
                assert external == []
            finally:
                browser.close()
