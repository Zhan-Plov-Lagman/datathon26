from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from app.api.schema import AnalysisResult
from app.jobs.store import JobStore, QueueError


def make_queue(settings):
    clock = [1000.0]
    return JobStore(settings, clock=lambda: clock[0]), clock


def test_simultaneous_first_start(settings):
    with ThreadPoolExecutor(max_workers=8) as pool:
        stores = list(pool.map(lambda _: JobStore(settings), range(8)))
    assert len(stores) == 8
    stores[0].create("shared-job")
    assert all(store.get("shared-job")["status"] == "processing" for store in stores)


def claim(store, worker="gpu-a"):
    store.hello(worker, True, "test")
    return store.claim(worker)


def test_restart_preserves_queue_and_result(settings, result):
    store, clock = make_queue(settings)
    store.create("one")
    restarted = JobStore(settings, clock=lambda: clock[0])
    lease = claim(restarted)
    assert restarted.get("one") == {"job_id": "one", "status": "processing"}
    restarted.complete("one", "gpu-a", lease["attempt_token"], AnalysisResult.model_validate(result))
    assert JobStore(settings, clock=lambda: clock[0]).get("one")["verdict"] == "violation"


def test_crashed_worker_reassigned_and_stale_result_rejected(settings, result):
    store, clock = make_queue(settings)
    store.create("one")
    first = claim(store)
    clock[0] += settings.lease_sec + 1
    store = JobStore(settings, clock=lambda: clock[0])
    second = claim(store, "gpu-b")
    assert second["attempt"] == 2
    assert second["attempt_token"] != first["attempt_token"]
    with pytest.raises(QueueError, match="no longer active"):
        store.complete("one", "gpu-a", first["attempt_token"], AnalysisResult.model_validate(result))
    store.complete("one", "gpu-b", second["attempt_token"], AnalysisResult.model_validate(result))
    with pytest.raises(QueueError):
        store.fail("one", "gpu-a", first["attempt_token"], "analysis_failed")
    assert store.get("one")["status"] == "done"


def test_retry_limit_and_deadline(settings):
    store, clock = make_queue(settings)
    store.create("one")
    claim(store)
    clock[0] += settings.lease_sec + 1
    claim(store, "gpu-b")
    clock[0] += settings.lease_sec + 1
    assert store.get("one") == {"job_id": "one", "status": "failed"}
    assert claim(store, "gpu-c") is None
    store.create("queued-forever")
    clock[0] += settings.job_timeout_sec
    assert store.get("queued-forever")["status"] == "failed"


def test_heartbeat_cannot_extend_overall_deadline_or_revive_expired_lease(settings):
    store, clock = make_queue(settings)
    store.create("one")
    lease = claim(store)
    for _ in range(7):
        clock[0] += 20
        store.heartbeat("one", "gpu-a", lease["attempt_token"])
    clock[0] += settings.lease_sec
    with pytest.raises(QueueError):
        store.heartbeat("one", "gpu-a", lease["attempt_token"])
    clock[0] = 1000 + settings.job_timeout_sec
    assert store.get("one")["status"] == "failed"


def test_atomic_claim_with_two_workers(settings):
    store, _ = make_queue(settings)
    for job in ["one", "two"]:
        store.create(job)
    for worker in ["gpu-a", "gpu-b"]:
        store.hello(worker, True, "test")
    with ThreadPoolExecutor(max_workers=2) as pool:
        leases = list(pool.map(store.claim, ["gpu-a", "gpu-b"]))
    assert {item["job_id"] for item in leases} == {"one", "two"}
    with pytest.raises(QueueError, match="already has"):
        store.claim("gpu-a")


def test_only_one_worker_can_claim_last_job(settings):
    store, _ = make_queue(settings)
    store.create("one")
    with ThreadPoolExecutor(max_workers=2) as pool:
        leases = list(pool.map(lambda worker: claim(store, worker), ["gpu-a", "gpu-b"]))
    assert sum(lease is not None for lease in leases) == 1


def test_failed_analysis_never_means_no_violation_and_next_job_runs(settings):
    store, _ = make_queue(settings)
    store.create("one")
    lease = claim(store)
    store.fail("one", "gpu-a", lease["attempt_token"], "analysis_failed")
    assert store.get("one") == {"job_id": "one", "status": "failed"}
    store.create("two")
    assert store.claim("gpu-a")["job_id"] == "two"


def test_explicit_retry_is_bounded(settings):
    store, _ = make_queue(settings)
    store.create("one")
    for _ in range(settings.max_attempts):
        lease = claim(store)
        store.fail("one", "gpu-a", lease["attempt_token"], "worker_unavailable", retryable=True)
    assert store.get("one")["status"] == "failed"


def test_queue_limit_and_worker_readiness_expiry(settings):
    store, clock = make_queue(replace(settings, max_pending_jobs=1))
    assert not store.readiness()["ready"]
    store.hello("gpu-a", True, "test")
    assert store.readiness()["ready"]
    clock[0] += settings.worker_ttl_sec
    assert not store.readiness()["ready"]
    with pytest.raises(QueueError):
        store.claim("gpu-a")
    store.create("one")
    with pytest.raises(QueueError) as error:
        store.create("two")
    assert error.value.status == 503
