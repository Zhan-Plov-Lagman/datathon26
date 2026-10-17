from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.api.server import create_public_app, create_worker_app
from app.jobs.store import JobStore


def clients(settings):
    store = JobStore(settings)
    return (TestClient(create_public_app(settings, store), headers={"X-API-Key": settings.api_key}),
            TestClient(create_worker_app(settings, store), headers={"X-Worker-Key": settings.worker_key}))


def upload(public, content=b"synthetic transport fixture, not actual video", filename="test.mov"):
    return public.post("/api/v1/analyze", files={"file": (filename, content, "video/quicktime")})


def claim(private):
    response = private.post("/internal/jobs/claim", json={"worker_id": "gpu-a", "ready": True, "pipeline_version": "test"})
    assert response.status_code == 200
    return response.json()


@pytest.mark.parametrize("key", [None, "wrong"])
@pytest.mark.parametrize("method,path", [("post", "/api/v1/analyze"), ("get", "/api/v1/jobs/unknown")])
def test_auth(settings, key, method, path):
    with TestClient(create_public_app(settings)) as client:
        response = getattr(client, method)(path, headers={} if key is None else {"X-API-Key": key})
        assert response.status_code == 401
    assert not list((settings.storage / "uploads").iterdir())


def test_full_public_contract_restart_and_private_isolation(settings, result):
    public, private = clients(settings)
    with public, private:
        assert public.get("/api/v1/readiness").status_code == 503
        response = upload(public, filename="../../EVIL.MOV")
        assert response.status_code == 202
        assert set(response.json()) == {"job_id"}
        job_id = response.json()["job_id"]
        assert public.get(f"/api/v1/jobs/{job_id}").json() == {"job_id": job_id, "status": "processing"}
        lease = claim(private)
        assert public.get("/api/v1/readiness").json()["ready"]
        assert public.post("/internal/jobs/claim").status_code == 404
        assert private.post("/internal/jobs/claim", headers={"X-Worker-Key": settings.api_key}).status_code == 401
        attempt = {"worker_id": "gpu-a", "attempt_token": lease["attempt_token"]}
        file_headers = {"X-Worker-Id": "gpu-a", "X-Attempt-Token": lease["attempt_token"]}
        assert private.get(lease["download_path"]).status_code == 409
        assert private.get(lease["download_path"], headers=file_headers).content.startswith(b"synthetic")
        assert private.post(f"/internal/jobs/{job_id}/heartbeat", json=attempt).status_code == 200
        assert private.post(f"/internal/jobs/{job_id}/complete", json={**attempt, "result": result}).status_code == 200
        assert private.post(f"/internal/jobs/{job_id}/complete", json={**attempt, "result": result}).status_code == 409
    with TestClient(create_public_app(settings), headers={"X-API-Key": settings.api_key}) as restarted:
        body = restarted.get(f"/api/v1/jobs/{job_id}").json()
        assert set(body) == {"job_id", "status", "verdict", "violations"}
        assert body["verdict"] == "violation"
        assert "rules_version" not in body


def test_empty_result_and_unknown_job(settings, result):
    result.update(verdict="no_violation", violations=[], rule_ids=[])
    public, private = clients(settings)
    with public, private:
        job_id = upload(public).json()["job_id"]
        lease = claim(private)
        response = private.post(f"/internal/jobs/{job_id}/complete", json={
            "worker_id": "gpu-a", "attempt_token": lease["attempt_token"], "result": result})
        assert response.status_code == 200
        assert public.get(f"/api/v1/jobs/{job_id}").json() == {
            "job_id": job_id, "status": "done", "verdict": "no_violation", "violations": []}
        assert public.get("/api/v1/jobs/unknown").status_code == 404


def test_simultaneous_http_uploads_do_not_mix_files(settings):
    public, private = clients(settings)
    with public, private:
        payloads = [b"synthetic-one", b"synthetic-two"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(lambda data: upload(public, content=data), payloads))
        assert all(response.status_code == 202 for response in responses)
        expected = {response.json()["job_id"]: data for response, data in zip(responses, payloads)}
        assert len(expected) == 2
        for worker in ["gpu-a", "gpu-b"]:
            lease = private.post("/internal/jobs/claim", json={
                "worker_id": worker, "ready": True, "pipeline_version": "test"}).json()
            data = private.get(lease["download_path"], headers={
                "X-Worker-Id": worker, "X-Attempt-Token": lease["attempt_token"]}).content
            assert data == expected[lease["job_id"]]


@pytest.mark.parametrize("filename,content,status", [("file.txt", b"test", 422), ("test.mov", b"", 422),
                                                       ("large.mov", b"x" * 1025, 413)])
def test_invalid_upload_cleanup(settings, filename, content, status):
    settings = replace(settings, max_upload_bytes=1024)
    public, _ = clients(settings)
    with public:
        assert upload(public, filename=filename, content=content).status_code == status
    assert not list((settings.storage / "uploads").iterdir())


def test_queue_full_removes_unaccepted_input(settings):
    settings = replace(settings, max_pending_jobs=1)
    public, _ = clients(settings)
    with public:
        assert upload(public).status_code == 202
        assert upload(public).status_code == 503
    assert len(list((settings.storage / "uploads").iterdir())) == 1


def test_streamed_size_limit_and_auth_before_body(settings):
    public, _ = clients(replace(settings, max_upload_bytes=10))
    def stream():
        for _ in range(70):
            yield b"a" * 1024
    with public:
        response = public.post("/api/v1/analyze", content=stream(), headers={
            "Content-Type": "multipart/form-data; boundary=x", "Transfer-Encoding": "chunked"})
        assert response.status_code == 413
        response = public.post("/api/v1/analyze", content=stream(), headers={"X-API-Key": "wrong"})
        assert response.status_code == 401


def test_invalid_result_redacted_and_job_can_fail(settings):
    public, private = clients(settings)
    with public, private:
        job_id = upload(public, content=b"corrupt MOV").json()["job_id"]
        lease = claim(private)
        attempt = {"worker_id": "gpu-a", "attempt_token": lease["attempt_token"]}
        response = private.post(f"/internal/jobs/{job_id}/complete", json={**attempt, "result": {
            "analysis_kind": "observations_only", "secret_scene": "DO_NOT_ECHO"}})
        assert response.status_code == 422
        assert "DO_NOT_ECHO" not in response.text
        assert private.post(f"/internal/jobs/{job_id}/fail", json={**attempt, "code": "invalid_video"}).status_code == 200
        assert public.get(f"/api/v1/jobs/{job_id}").json() == {"job_id": job_id, "status": "failed"}


def test_static_assets_are_local_and_no_content_routes(settings):
    public, _ = clients(settings)
    with public:
        page = public.get("/")
        assert page.status_code == 200
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
        assert public.get("/assets/app.js").status_code == 200
        assert public.get("/assets/style.css").status_code == 200
        assert public.get("/uploads/test.mov").status_code == 404
        assert public.get("/docs").status_code == 404
        assert public.get("/healthz").json() == {"status": "ok"}
