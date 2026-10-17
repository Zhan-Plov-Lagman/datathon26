"""Run the coordinator and one real GPU worker on the same laptop."""
from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time

import httpx
from dotenv import load_dotenv

from app.api.settings import Settings


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=8)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    if os.name == "nt":
        # Only the tree started by this supervisor, including the GPU child.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, timeout=10)
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=10)


def require_free_ports():
    with contextlib.ExitStack() as stack:
        for port in (8000, 8001):
            probe = stack.enter_context(socket.socket())
            probe.bind(("127.0.0.1", port))


def wait_for_coordinator(process, timeout=20):
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=.5, trust_env=False) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("coordinator_exited")
            try:
                public = client.get("http://127.0.0.1:8000/healthz")
                private = client.post("http://127.0.0.1:8001/internal/jobs/claim")
                if public.status_code == 200 and private.status_code == 401:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(.1)
    raise RuntimeError("coordinator_startup_timeout")


def main():
    load_dotenv(override=False)
    try:
        Settings.from_env()
    except (ValueError, TypeError):
        print("BLOCKED: invalid coordinator settings; check local .env without sharing keys.")
        return 2
    if not os.environ.get("FORTE_RULES_FACTORY"):
        print("BLOCKED: FORTE_RULES_FACTORY missing; A/B facts and reviewed evaluator are required.")
        return 2
    try:
        require_free_ports()
    except OSError:
        print("BLOCKED: local ports 8000/8001 unavailable; check the existing coordinator first.")
        return 2
    children = []
    options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
               else {"start_new_session": True})
    try:
        coordinator = subprocess.Popen([sys.executable, "-m", "app.api"], **options)
        children.append(coordinator)
        wait_for_coordinator(coordinator)
        worker = subprocess.Popen([sys.executable, "-m", "app.jobs.worker", "--worker-id", "gpu-a",
                                   "--coordinator", "http://127.0.0.1:8001"], **options)
        children.append(worker)
        print("UI: http://127.0.0.1:8000; worker is loading. Check readiness before uploading.", flush=True)
        while True:
            if any(child.poll() is not None for child in children):
                print("STOPPED: a service exited. No analysis readiness is assumed.", flush=True)
                return 1
            time.sleep(.2)
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError):
        print("STOPPED: local startup failed; no video or exception details printed.", flush=True)
        return 1
    finally:
        for child in reversed(children):
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                stop_process(child)


if __name__ == "__main__":
    raise SystemExit(main())
