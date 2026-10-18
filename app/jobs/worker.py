"""Sequential worker transport. Persistent GPU child, external timeout, private HTTP only."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing
import os
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv

from app.api.schema import AnalysisResult
from app.jobs.pipeline import import_factory


def _child(pipe, factory):
    # Model libraries sometimes print model output: silence it in the transport process.
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            processor = import_factory(factory)()
            pipe.send({"ready": True, "pipeline_version": processor.pipeline_version})
        except Exception:
            pipe.send({"ready": False})
            return
        while True:
            try:
                task = pipe.recv()
            except EOFError:
                return
            try:
                result = processor.process(task["video"], task["job_id"], task["timeout_sec"])
                validated = AnalysisResult.model_validate(result)
                Path(task["output"]).write_text(validated.model_dump_json(), encoding="utf-8")
                pipe.send({"ok": True})
            except Exception:
                pipe.send({"ok": False})


class ProcessorProcess:
    def __init__(self, factory, startup_timeout=180):
        context = multiprocessing.get_context("spawn")
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=_child, args=(child, factory), daemon=True)
        self.process.start()
        child.close()
        try:
            if not self.pipe.poll(startup_timeout):
                raise RuntimeError("Pipeline startup timed out")
            hello = self.pipe.recv()
            if not hello["ready"]:
                raise RuntimeError("Pipeline unavailable: check local models and configured detector release")
            self.version = hello["pipeline_version"]
        except BaseException:
            self.close()
            raise

    def run(self, task, deadline, lost):
        self.pipe.send(task)
        while time.monotonic() < deadline and not lost.is_set():
            if self.pipe.poll(min(0.2, max(0, deadline - time.monotonic()))):
                return self.pipe.recv()["ok"]
        self.close()  # Cooperative model deadlines cannot protect a stuck CUDA process.
        raise TimeoutError

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=5)
        self.pipe.close()


class LeaseHeartbeat:
    def __init__(self, client, path, attempt, interval):
        self.client, self.path, self.attempt, self.interval = client, path, attempt, interval
        self.stop, self.lost = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.wait(self.interval):
            try:
                self.client.post(self.path + "/heartbeat", json=self.attempt).raise_for_status()
            except httpx.HTTPError:
                self.lost.set()
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=6)


def execute_job(client, processor, lease, worker_id, work_dir, claimed_at, max_download_bytes=256 * 1024 * 1024):
    attempt = {"worker_id": worker_id, "attempt_token": lease["attempt_token"]}
    path = f"/internal/jobs/{lease['job_id']}"
    deadline = claimed_at + lease["remaining_sec"] - 1  # Reserve time for completion request.
    with TemporaryDirectory(prefix="job-", dir=work_dir) as directory:
        video, output = Path(directory) / "input.mov", Path(directory) / "result.json"
        with LeaseHeartbeat(client, path, attempt, min(10, lease["lease_sec"] / 3)) as heartbeat:
            try:
                with client.stream("GET", lease["download_path"], headers={
                    "X-Worker-Id": worker_id, "X-Attempt-Token": lease["attempt_token"],
                }) as response:
                    response.raise_for_status()
                    size = 0
                    with video.open("xb") as handle:
                        for chunk in response.iter_bytes(1024 * 1024):
                            size += len(chunk)
                            if size > max_download_bytes:
                                raise ValueError("Input exceeds worker limit")
                            if time.monotonic() >= deadline or heartbeat.lost.is_set():
                                raise TimeoutError
                            handle.write(chunk)
                timeout = deadline - time.monotonic()
                if timeout <= 0 or heartbeat.lost.is_set():
                    raise TimeoutError
                ok = processor.run({"video": str(video), "output": str(output),
                                    "job_id": lease["job_id"], "timeout_sec": timeout}, deadline, heartbeat.lost)
                if not ok:
                    raise ValueError("Analysis failed")
                result = json.loads(output.read_text(encoding="utf-8"))
                client.post(path + "/complete", json={**attempt, "result": result}).raise_for_status()
                return True
            except (httpx.HTTPError, OSError, ValueError, TimeoutError, EOFError):
                # Only enumerated technical codes leave this process, never exception text.
                with contextlib.suppress(httpx.HTTPError):
                    client.post(path + "/fail", json={**attempt, "code": "analysis_failed", "retryable": False})
                return False


def main():
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", default=os.environ.get("FORTE_COORDINATOR", "http://127.0.0.1:8001"))
    parser.add_argument("--worker-id", default=os.environ.get("FORTE_WORKER_ID", "gpu-a"))
    parser.add_argument("--factory", default=os.environ.get('FORTE_PROCESSOR_FACTORY','app.jobs.pipeline:create_processor'))
    parser.add_argument('--startup-timeout',type=float,default=os.environ.get('FORTE_STARTUP_TIMEOUT','180'))
    parser.add_argument("--work-dir", type=Path, default=Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local/share"))) / "Datathon26/worker")
    args = parser.parse_args()
    if not math.isfinite(args.startup_timeout) or args.startup_timeout<=0:
        parser.error('Startup timeout must be positive and finite')
    url = urlsplit(args.coordinator)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.query or url.fragment:
        parser.error("Use a private coordinator HTTP(S) URL without credentials or query")
    key = os.environ.get("FORTE_WORKER_KEY", "")
    if len(key) < 32:
        parser.error("Set FORTE_WORKER_KEY (at least 32 characters)")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    processor = None
    with httpx.Client(base_url=args.coordinator, headers={"X-Worker-Key": key},
                      timeout=5, trust_env=False, follow_redirects=False) as client:
        try:
            while True:
                if processor is None or not processor.process.is_alive():
                    if processor:
                        processor.close()
                    processor = ProcessorProcess(args.factory,startup_timeout=args.startup_timeout)
                hello = {"worker_id": args.worker_id, "ready": True, "pipeline_version": processor.version}
                try:
                    claimed_at = time.monotonic()
                    response = client.post("/internal/jobs/claim", json=hello)
                    response.raise_for_status()
                    if response.status_code == 204:
                        time.sleep(2)
                        continue
                    ok = execute_job(client, processor, response.json(), args.worker_id, args.work_dir, claimed_at)
                    print("job_done" if ok else "job_failed", flush=True)
                    if not ok:
                        processor.close()
                except httpx.HTTPError:
                    print("coordinator_unavailable", flush=True)
                    time.sleep(2)
        except KeyboardInterrupt:
            pass
        except RuntimeError as exc:
            # Our startup error has fixed text and contains no model response.
            print(str(exc), flush=True)
            raise SystemExit(1)
        finally:
            if processor:
                with contextlib.suppress(httpx.HTTPError):
                    client.post("/internal/workers/heartbeat", json={"worker_id": args.worker_id,
                                "ready": False, "pipeline_version": processor.version})
                processor.close()


if __name__ == "__main__":
    main()
