"""Submit a local MOV to the team's API; print technical aggregates only."""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

from app.api.schema import Violation


class CheckFailed(Exception):
    """Fixed error codes only; never include response, filename, or scene text."""


def api_url(value):
    url = urlsplit(value)
    if (url.scheme not in ("http", "https") or not url.hostname or url.username or url.password
            or url.query or url.fragment or url.path not in ("", "/")):
        raise argparse.ArgumentTypeError("Use a team HTTPS origin or local HTTP origin")
    if url.scheme == "http":
        local = url.hostname.lower() == "localhost"
        try:
            local = local or ipaddress.ip_address(url.hostname).is_loopback
        except ValueError:
            pass
        if not local:
            raise argparse.ArgumentTypeError("Remote API must use HTTPS")
    try:
        url.port
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid API port") from None
    return value.rstrip("/")


def response_json(response, expected):
    if response.status_code != expected:
        raise CheckFailed("unexpected_http_status")
    try:
        body = response.json()
    except ValueError:
        raise CheckFailed("invalid_json") from None
    if not isinstance(body, dict):
        raise CheckFailed("invalid_schema")
    return body


def validate_done(body):
    if (set(body) != {"job_id", "status", "verdict", "violations"}
            or body["verdict"] not in ("violation", "no_violation")
            or not isinstance(body["violations"], list) or len(body["violations"]) > 100
            or (body["verdict"] == "violation") != bool(body["violations"])):
        raise CheckFailed("invalid_result_schema")
    try:
        events = [Violation.model_validate(event) for event in body["violations"]]
    except ValidationError:
        raise CheckFailed("invalid_evidence_or_event") from None
    return len(events), sum(len(event.evidence_frames) for event in events)


def run_check(client, video, timeout=300, interval=2):
    started = time.monotonic()
    deadline = started + timeout

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise CheckFailed("deadline_exceeded")
        return value

    readiness = client.get("/api/v1/readiness", timeout=remaining())
    if readiness.status_code == 503:
        raise CheckFailed("analysis_not_ready")
    ready = response_json(readiness, 200)
    if ready.get("ready") is not True:
        raise CheckFailed("analysis_not_ready")
    with video.open("rb") as handle:
        uploaded = response_json(client.post("/api/v1/analyze", timeout=remaining(),
            files={"file": ("input.mov", handle, "video/quicktime")}), 202)
    job_id = uploaded.get("job_id")
    if set(uploaded) != {"job_id"} or not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
        raise CheckFailed("invalid_job_id")
    while True:
        body = response_json(client.get(f"/api/v1/jobs/{job_id}", timeout=remaining()), 200)
        remaining()  # The response itself must arrive within the total budget.
        if body.get("job_id") != job_id:
            raise CheckFailed("job_id_mismatch")
        if body.get("status") == "failed":
            raise CheckFailed("analysis_failed")
        if body.get("status") == "done":
            count, frames = validate_done(body)
            remaining()
            return {"completed": True, "elapsed_sec": round(time.monotonic() - started, 3),
                    "events_count": count, "jpeg_frames_checked": frames}
        if set(body) != {"job_id", "status"} or body.get("status") != "processing":
            raise CheckFailed("invalid_status_schema")
        time.sleep(min(interval, remaining()))


def main():
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", type=api_url)
    parser.add_argument("video", type=Path)
    args = parser.parse_args()
    key = os.environ.get("FORTE_API_KEY", "")
    if len(key) < 32:
        print('{"completed": false, "error": "api_key_missing"}')
        return 2
    if args.video.suffix.lower() != ".mov" or not args.video.is_file():
        print('{"completed": false, "error": "local_mov_required"}')
        return 2
    try:
        with httpx.Client(base_url=args.base_url, headers={"X-API-Key": key}, trust_env=False,
                          follow_redirects=False) as client:
            print(json.dumps(run_check(client, args.video)))
        return 0
    except CheckFailed as exc:
        print(json.dumps({"completed": False, "error": str(exc)}))
    except (httpx.HTTPError, OSError, ValueError):
        print('{"completed": false, "error": "transport_or_local_io_error"}')
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
