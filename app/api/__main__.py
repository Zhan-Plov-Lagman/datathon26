"""Run two separately bound servers; only port 8000 goes to the public proxy."""
import argparse
import multiprocessing
import time

from dotenv import load_dotenv

from app.api.launcher import serve
from app.api.settings import Settings
from app.jobs.store import JobStore


def main():
    load_dotenv(override=False)
    JobStore(Settings.from_env())  # Validate settings and initialize SQLite before listeners.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-host", default="127.0.0.1")
    parser.add_argument("--public-port", type=int, default=8000)
    parser.add_argument("--worker-host", default="127.0.0.1")
    parser.add_argument("--worker-port", type=int, default=8001)
    args = parser.parse_args()
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=serve, args=("app.api.server:create_public_app", args.public_host, args.public_port)),
        context.Process(target=serve, args=("app.api.server:create_worker_app", args.worker_host, args.worker_port)),
    ]
    failed = False
    try:
        for process in processes:
            process.start()
        print(f"UI/API: http://{args.public_host}:{args.public_port}; private worker port: {args.worker_port}", flush=True)
        while all(process.is_alive() for process in processes):
            time.sleep(0.5)
        failed = True
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid:
                process.join(timeout=5)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
