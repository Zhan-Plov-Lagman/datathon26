"""Synthetic-only GPU smoke test; no bank frames or labels are read."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from app.inference.pipeline import LocalVideoAnalyzer
from app.inference.qwen import QwenObserver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/Qwen3-VL-2B-Instruct-shards")
    args = parser.parse_args()
    folder = Path("artifacts/synthetic-smoke")
    folder.mkdir(parents=True, exist_ok=True)
    video = folder / "generated.mov"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=320x192:rate=10:duration=3", "-c:v", "libx264", str(video)],
                   check=True, capture_output=True, timeout=30)
    started = time.monotonic()
    observer = QwenObserver(args.model_dir)
    load_sec = time.monotonic() - started
    analyzer = LocalVideoAnalyzer(observer, frames_per_window=4)
    report = analyzer.analyze(video, "synthetic-smoke", timeout_sec=120)
    report["metrics"]["model_load_sec"] = round(load_sec, 3)
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"synthetic_smoke_passed": True, "metrics": report["metrics"]}))


if __name__ == "__main__":
    main()
