from copy import deepcopy

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from app.api.schema import AnalysisResult
from app.api.server import create_public_app, create_worker_app
from app.jobs.store import JobStore


def signs(result):
    body = deepcopy(result)
    body.update(analysis_kind="signs_evaluated", rules_reviewed=False,
                source_texts_verified=True, accuracy_status="unmeasured",
                negative_policy_version="signs-targeted.v1")
    return body


@pytest.mark.parametrize("positive", [True, False])
def test_signs_completion_survives_restart_with_exact_public_contract(settings, result, positive):
    body = signs(result)
    if not positive:
        body.update(verdict="no_violation", violations=[], rule_ids=[])
    store = JobStore(settings)
    with TestClient(create_public_app(settings, store), headers={"X-API-Key": settings.api_key}) as public, TestClient(create_worker_app(settings, store), headers={"X-Worker-Key": settings.worker_key}) as private:
        uploaded = public.post("/api/v1/analyze", files={"file": ("synthetic.mov", b"synthetic transport fixture", "video/quicktime")})
        assert uploaded.status_code == 202
        job_id = uploaded.json()["job_id"]
        lease = private.post("/internal/jobs/claim", json={"worker_id": "signs-test", "ready": True, "pipeline_version": "synthetic"}).json()
        completed = private.post(f"/internal/jobs/{job_id}/complete", json={"worker_id": "signs-test", "attempt_token": lease["attempt_token"], "result": body})
        assert completed.status_code == 200
    with TestClient(create_public_app(settings), headers={"X-API-Key": settings.api_key}) as restarted:
        actual = restarted.get(f"/api/v1/jobs/{job_id}").json()
        assert set(actual) == {"job_id", "status", "verdict", "violations"}
        assert actual == AnalysisResult.model_validate(body).public(job_id)


@pytest.mark.parametrize("field,value", [("rules_reviewed", True), ("source_texts_verified", False), ("source_texts_verified", None), ("accuracy_status", None), ("negative_policy_version", "other"), ("complete", False), ("analysis_kind", "observations_only")])
def test_signs_requires_honest_complete_metadata(result, field, value):
    body = signs(result)
    body[field] = value
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(body)


def test_original_reviewed_path_still_requires_review(result):
    AnalysisResult.model_validate(result)
    result["rules_reviewed"] = False
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(result)


def test_incomplete_signs_cannot_be_successful_negative(settings, result):
    body = signs(result)
    body.update(complete=False, verdict="no_violation", violations=[], rule_ids=[])
    store = JobStore(settings)
    with TestClient(create_public_app(settings, store), headers={"X-API-Key": settings.api_key}) as public, TestClient(create_worker_app(settings, store), headers={"X-Worker-Key": settings.worker_key}) as private:
        job_id = public.post("/api/v1/analyze", files={"file": ("synthetic.mov", b"synthetic", "video/quicktime")}).json()["job_id"]
        lease = private.post("/internal/jobs/claim", json={"worker_id": "signs-test", "ready": True, "pipeline_version": "synthetic"}).json()
        attempt = {"worker_id": "signs-test", "attempt_token": lease["attempt_token"]}
        assert private.post(f"/internal/jobs/{job_id}/complete", json={**attempt, "result": body}).status_code == 422
        assert private.post(f"/internal/jobs/{job_id}/fail", json={**attempt, "code": "analysis_failed"}).status_code == 200
        assert public.get(f"/api/v1/jobs/{job_id}").json() == {"job_id": job_id, "status": "failed"}
