from __future__ import annotations

import asyncio
import os
import secrets
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.responses import FileResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from app.api.schema import Attempt, Completion, Failure, WorkerHello
from app.api.settings import Settings
from app.jobs.store import JobStore, QueueError

WEB = Path(__file__).resolve().parents[2] / "web"


class RequestGuard:
    """Authenticate before parsing/spooling uploads; cap streamed and chunked bodies."""
    def __init__(self, app, key, private, max_body):
        self.app, self.key, self.private, self.max_body = app, key.encode(), private, max_body

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        protected = self.private or scope["path"].startswith("/api/")
        if protected:
            name = b"x-worker-key" if self.private else b"x-api-key"
            if not secrets.compare_digest(headers.get(name, b""), self.key):
                return await JSONResponse({"detail": "Unauthorized"}, 401)(scope, receive, send)
        try:
            length = int(headers.get(b"content-length", b"0"))
            if length < 0:
                raise ValueError
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, 400)(scope, receive, send)
        if length > self.max_body:
            return await JSONResponse({"detail": "Request too large"}, 413)(scope, receive, send)
        scope["upload_started"] = time.time()
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self.max_body:
                raise HTTPException(413, "Request too large")
            return message

        async def safe_send(message):
            if message["type"] == "http.response.start":
                message["headers"] += [
                    (b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"content-security-policy", b"default-src 'self'; script-src 'self'; style-src 'self'; "
                     b"img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; "
                     b"object-src 'none'; base-uri 'none'; frame-ancestors 'none'"),
                ]
            await send(message)

        await self.app(scope, limited_receive, safe_send)


def _base(settings, store, private):
    @asynccontextmanager
    async def lifespan(app):
        async def sweep():
            while True:
                await run_in_threadpool(store.reap)
                await asyncio.sleep(1)
        task = asyncio.create_task(sweep())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="ForteBank worker transport" if private else "ForteBank API",
                  docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.store = store
    app.add_middleware(RequestGuard, key=settings.worker_key if private else settings.api_key,
                       private=private, max_body=settings.max_result_bytes if private else settings.max_upload_bytes + 65536)

    @app.exception_handler(QueueError)
    async def queue_error(request, exc):
        return JSONResponse({"detail": exc.detail}, exc.status)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Validation errors may include image data, keys or descriptions: never echo them.
        return JSONResponse({"detail": "Invalid request schema"}, 422)

    @app.exception_handler(OSError)
    @app.exception_handler(sqlite3.Error)
    async def storage_error(request, exc):
        return JSONResponse({"detail": "Local storage unavailable"}, 503)

    return app


def _save_upload(upload, job_id, settings, store, started):
    partial = store.uploads / f"{job_id}.part"
    target = store.uploads / f"{job_id}.mov"
    created = False
    try:
        size = 0
        with partial.open("xb") as output:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(413, "Video too large")
                output.write(chunk)
            if size == 0:
                raise HTTPException(422, "Video is empty")
            output.flush()
            os.fsync(output.fileno())
        os.replace(partial, target)
        if os.name != "nt":
            fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        store.create(job_id, started=started)
        created = True
    finally:
        partial.unlink(missing_ok=True)
        if not created:
            target.unlink(missing_ok=True)


def create_public_app(settings: Settings | None = None, store=None):
    settings = settings or Settings.from_env()
    store = store or JobStore(settings)
    app = _base(settings, store, False)

    @app.post("/api/v1/analyze", status_code=202)
    async def analyze(request: Request):
        async with request.form(max_files=1, max_fields=0, max_part_size=1024) as form:
            file = form.get("file")
            if not isinstance(file, UploadFile) or Path(file.filename or "").suffix.lower() != ".mov":
                raise HTTPException(422, "Expected MOV video in multipart field file")
            job_id = uuid.uuid4().hex
            await run_in_threadpool(_save_upload, file, job_id, settings, store, request.scope["upload_started"])
        return {"job_id": job_id}

    @app.get("/api/v1/jobs/{job_id}")
    def job(job_id: str):
        return store.get(job_id)

    @app.get("/api/v1/readiness")
    def readiness():
        state = store.readiness()
        return JSONResponse(state, 200 if state["ready"] else 503)

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(WEB / "index.html")

    app.mount("/assets", StaticFiles(directory=WEB), name="assets")
    return app


def create_worker_app(settings: Settings | None = None, store=None):
    settings = settings or Settings.from_env()
    store = store or JobStore(settings)
    app = _base(settings, store, True)

    @app.post("/internal/workers/heartbeat")
    def hello(body: WorkerHello):
        store.hello(body.worker_id, body.ready, body.pipeline_version)
        return {"ok": True}

    @app.post("/internal/jobs/claim")
    def claim(body: WorkerHello):
        store.hello(body.worker_id, body.ready, body.pipeline_version)
        result = store.claim(body.worker_id)
        return Response(status_code=204) if result is None else result

    @app.get("/internal/jobs/{job_id}/file")
    def file(job_id: str, request: Request):
        path = store.file(job_id, request.headers.get("x-worker-id", ""),
                          request.headers.get("x-attempt-token", ""))
        return FileResponse(path, media_type="video/quicktime", filename="input.mov")

    @app.post("/internal/jobs/{job_id}/heartbeat")
    def heartbeat(job_id: str, body: Attempt):
        return store.heartbeat(job_id, body.worker_id, body.attempt_token)

    @app.post("/internal/jobs/{job_id}/complete")
    def complete(job_id: str, body: Completion):
        store.complete(job_id, body.worker_id, body.attempt_token, body.result)
        return {"ok": True}

    @app.post("/internal/jobs/{job_id}/fail")
    def fail(job_id: str, body: Failure):
        store.fail(job_id, body.worker_id, body.attempt_token, body.code, body.retryable)
        return {"ok": True}

    return app
