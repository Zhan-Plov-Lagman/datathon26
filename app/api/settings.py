from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    storage: Path
    api_key: str
    worker_key: str
    max_upload_bytes: int = 256 * 1024 * 1024
    max_result_bytes: int = 32 * 1024 * 1024
    job_timeout_sec: int = 240
    lease_sec: int = 30
    worker_ttl_sec: int = 45
    max_attempts: int = 2
    max_pending_jobs: int = 8

    def __post_init__(self):
        if min(len(self.api_key), len(self.worker_key)) < 32 or self.api_key == self.worker_key:
            raise ValueError("Set distinct FORTE_API_KEY and FORTE_WORKER_KEY (at least 32 characters)")
        for name in ("max_upload_bytes", "max_result_bytes", "job_timeout_sec", "lease_sec",
                     "worker_ttl_sec", "max_attempts", "max_pending_jobs"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.lease_sec < 6:
            raise ValueError("lease_sec must be at least 6")

    @classmethod
    def from_env(cls):
        # Keep SQLite and sensitive runtime files out of OneDrive and Git by default.
        root = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local/share")))
        kwargs = {"storage": Path(os.environ.get("FORTE_STORAGE", root / "Datathon26/runtime")),
                  "api_key": os.environ.get("FORTE_API_KEY", ""),
                  "worker_key": os.environ.get("FORTE_WORKER_KEY", "")}
        for name in ("max_upload_bytes", "max_result_bytes", "job_timeout_sec", "lease_sec",
                     "worker_ttl_sec", "max_attempts", "max_pending_jobs"):
            value = os.environ.get("FORTE_" + name.upper())
            if value is not None:
                kwargs[name] = int(value)
        return cls(**kwargs)
