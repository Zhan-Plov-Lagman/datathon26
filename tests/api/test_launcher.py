import os
import signal
import socket
import subprocess
import sys
import time

import httpx


def test_python_module_launcher_starts_both_listeners(settings):
    sockets = [socket.socket(), socket.socket()]
    for item in sockets:
        item.bind(("127.0.0.1", 0))
    public_port, private_port = [item.getsockname()[1] for item in sockets]
    for item in sockets:
        item.close()
    env = {**os.environ, "FORTE_STORAGE": str(settings.storage),
           "FORTE_API_KEY": settings.api_key, "FORTE_WORKER_KEY": settings.worker_key}
    kwargs = {"start_new_session": True} if os.name != "nt" else {}
    process = subprocess.Popen([sys.executable, "-m", "app.api", "--public-port", str(public_port),
                                "--worker-port", str(private_port)], env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
    try:
        with httpx.Client(timeout=.5, trust_env=False) as client:
            deadline = time.monotonic() + 15
            while True:
                assert process.poll() is None, "Coordinator launcher exited before startup"
                try:
                    if (client.get(f"http://127.0.0.1:{public_port}/healthz").status_code == 200
                            and client.post(f"http://127.0.0.1:{private_port}/internal/jobs/claim").status_code == 401):
                        break
                except httpx.HTTPError:
                    pass
                assert time.monotonic() < deadline, "Coordinator did not start"
                time.sleep(.05)
            assert client.post(f"http://127.0.0.1:{private_port}/internal/jobs/claim").status_code == 401
            assert client.post(f"http://127.0.0.1:{public_port}/internal/jobs/claim").status_code == 404
    finally:
        if os.name == "nt":
            # Only the subprocess tree started by this test, never other Python processes.
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=10)
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
