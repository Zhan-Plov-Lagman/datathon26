from __future__ import annotations

import json
import secrets
import sqlite3
import time
from contextlib import contextmanager

from app.api.schema import AnalysisResult
from app.api.settings import Settings


class QueueError(Exception):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail


class JobStore:
    def __init__(self, settings: Settings, clock=time.time):
        self.settings, self.clock = settings, clock
        settings.storage.mkdir(parents=True, exist_ok=True)
        self.uploads = settings.storage / "uploads"
        self.uploads.mkdir(exist_ok=True)
        self.database = settings.storage / "jobs.sqlite3"
        # Switching a fresh SQLite file to WAL may ignore busy_timeout when two
        # API processes bootstrap together. Retry only this idempotent setup.
        until = time.monotonic() + 10
        while True:
            try:
                self._initialize()
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or time.monotonic() >= until:
                    raise
                time.sleep(0.05)

    def _initialize(self):
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL, created REAL NOT NULL,
                    deadline REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT, token TEXT, lease_until REAL,
                    result TEXT, error_code TEXT
                );
                CREATE INDEX IF NOT EXISTS queue_order ON jobs(status, created);
                CREATE TABLE IF NOT EXISTS workers (
                    id TEXT PRIMARY KEY, seen REAL NOT NULL, ready INTEGER NOT NULL,
                    pipeline_version TEXT NOT NULL
                );
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.database, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def _reap(self, db, now):
        db.execute("""UPDATE jobs SET status='failed', error_code='deadline_exceeded',
                      token=NULL WHERE status IN ('queued','running') AND deadline<=?""", (now,))
        db.execute("""UPDATE jobs SET status=CASE WHEN attempts>=? THEN 'failed' ELSE 'queued' END,
                      error_code='lease_expired', token=NULL, worker_id=NULL, lease_until=NULL
                      WHERE status='running' AND lease_until<=?""", (self.settings.max_attempts, now))

    def reap(self):
        with self.transaction() as db:
            self._reap(db, self.clock())

    def create(self, job_id, started=None):
        now = self.clock()
        with self.transaction() as db:
            self._reap(db, now)
            count = db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if count >= self.settings.max_pending_jobs:
                raise QueueError(503, "Queue is full; retry later")
            db.execute("INSERT INTO jobs(id,status,created,deadline) VALUES(?,'queued',?,?)",
                       (job_id, now, (now if started is None else started) + self.settings.job_timeout_sec))

    def get(self, job_id):
        self.reap()
        with self.connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise QueueError(404, "Job not found")
        if row["status"] == "done":
            result = json.loads(row["result"])
            return {"job_id": job_id, "status": "done", "verdict": result["verdict"],
                    "violations": result["violations"]}
        return {"job_id": job_id, "status": "failed" if row["status"] == "failed" else "processing"}

    def hello(self, worker_id, ready, pipeline_version):
        with self.transaction() as db:
            db.execute("""INSERT INTO workers VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                          seen=excluded.seen, ready=excluded.ready, pipeline_version=excluded.pipeline_version""",
                       (worker_id, self.clock(), int(ready), pipeline_version))

    def readiness(self):
        with self.connection() as db:
            n = db.execute("SELECT count(*) FROM workers WHERE ready=1 AND seen>?",
                           (self.clock() - self.settings.worker_ttl_sec,)).fetchone()[0]
        return {"ready": n > 0, "ready_workers": n,
                "reason": None if n else "No ready analysis worker with reviewed rules"}

    def claim(self, worker_id):
        now = self.clock()
        with self.transaction() as db:
            self._reap(db, now)
            worker = db.execute("SELECT * FROM workers WHERE id=? AND ready=1 AND seen>?",
                                (worker_id, now - self.settings.worker_ttl_sec)).fetchone()
            if not worker:
                raise QueueError(409, "Worker is not ready")
            if db.execute("SELECT 1 FROM jobs WHERE status='running' AND worker_id=?", (worker_id,)).fetchone():
                raise QueueError(409, "Worker already has a job")
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created,id LIMIT 1").fetchone()
            if row is None:
                return None
            token = secrets.token_urlsafe(32)
            lease = min(now + self.settings.lease_sec, row["deadline"])
            db.execute("""UPDATE jobs SET status='running',worker_id=?,token=?,lease_until=?,
                          attempts=attempts+1,error_code=NULL WHERE id=?""", (worker_id, token, lease, row["id"]))
            return {"job_id": row["id"], "attempt_token": token, "attempt": row["attempts"] + 1,
                    "remaining_sec": row["deadline"] - now, "lease_sec": self.settings.lease_sec,
                    "download_path": f"/internal/jobs/{row['id']}/file"}

    def _active(self, db, job_id, worker_id, token, now):
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise QueueError(404, "Job not found")
        if (row["status"] != "running" or row["worker_id"] != worker_id or
                not secrets.compare_digest(row["token"] or "", token) or
                row["lease_until"] <= now or row["deadline"] <= now):
            raise QueueError(409, "Attempt is no longer active")
        return row

    def file(self, job_id, worker_id, token):
        with self.connection() as db:
            self._active(db, job_id, worker_id, token, self.clock())
        path = self.uploads / f"{job_id}.mov"
        if not path.is_file():
            raise QueueError(410, "Input file unavailable")
        return path

    def heartbeat(self, job_id, worker_id, token):
        now = self.clock()
        with self.transaction() as db:
            row = self._active(db, job_id, worker_id, token, now)
            db.execute("UPDATE jobs SET lease_until=? WHERE id=?",
                       (min(now + self.settings.lease_sec, row["deadline"]), job_id))
            db.execute("UPDATE workers SET seen=? WHERE id=?", (now, worker_id))
        return {"remaining_sec": row["deadline"] - now}

    def complete(self, job_id, worker_id, token, result: AnalysisResult):
        serialized = result.model_dump_json()
        with self.transaction() as db:
            self._active(db, job_id, worker_id, token, self.clock())
            db.execute("UPDATE jobs SET status='done',result=?,token=NULL WHERE id=?", (serialized, job_id))

    def fail(self, job_id, worker_id, token, code, retryable=False):
        now = self.clock()
        with self.transaction() as db:
            row = self._active(db, job_id, worker_id, token, now)
            retry = retryable and row["attempts"] < self.settings.max_attempts and row["deadline"] > now
            db.execute("UPDATE jobs SET status=?,error_code=?,token=NULL,worker_id=NULL WHERE id=?",
                       ("queued" if retry else "failed", code, job_id))
