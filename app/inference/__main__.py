from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline local observations; no legal verdict")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--timeout-sec", type=float, default=240)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--max-long-edge", type=int, default=560)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--start-sec", type=float, default=0)
    parser.add_argument("--end-sec", type=float)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("Output exists; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    job_id = args.job_id or uuid.uuid4().hex
    load_start = time.monotonic()
    try:
        from app.inference.qwen import QwenObserver
        from app.inference.pipeline import LocalVideoAnalyzer
        observer = QwenObserver(args.model_dir, max_new_tokens=args.max_new_tokens)
        load_sec = time.monotonic() - load_start
        analyzer = LocalVideoAnalyzer(observer, frames_per_window=args.frames, max_long_edge=args.max_long_edge)
        report = analyzer.analyze(args.video, job_id, args.timeout_sec, args.start_sec, args.end_sec)
        report["metrics"]["model_load_sec"] = round(load_sec, 3)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)
        # Bank observations are written locally, never printed to a coding assistant.
        print(json.dumps({"complete": True, "windows": len(report["windows"]),
                          "metrics": report["metrics"]}, ensure_ascii=True))
        return 0
    except Exception as exc:
        failure = {"schema_version": "observations.v1", "job_id": job_id,
                   "analysis_kind": "observations_only", "complete": False,
                   "error_type": type(exc).__name__}
        output.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(json.dumps({"complete": False, "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
