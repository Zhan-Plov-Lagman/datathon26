"""Local benchmark; stdout contains technical metrics, never video descriptions."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.inference.worker import LocalGPUWorker
from app.inference.qwen import QwenObserver
from app.video.sampling import probe_video


def main() -> None:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video-dir")
    inputs.add_argument("--video", help="Benchmark one explicit local MOV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", default="models/Qwen3-VL-2B-Instruct-shards")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--refine", action="store_true", help="Also test a middle interval, not a labeled event")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    # Select by duration only, without reading labels or inferring from filenames.
    paths = [Path(args.video)] if args.video else list(Path(args.video_dir).glob("*.mov"))
    if not paths:
        raise ValueError("No MOV inputs found")
    ordered = sorted(((p, probe_video(p)) for p in paths), key=lambda item: item[1].duration_sec)
    selected = [ordered[0]] if len(ordered) == 1 else [ordered[0], ordered[-1]]
    started = time.monotonic()
    observer = QwenObserver(args.model_dir)
    load_sec = round(time.monotonic() - started, 3)
    worker = LocalGPUWorker(observer)
    metrics = []
    for index, (path, info) in enumerate(selected):
        started = time.monotonic()
        row = {"sample_index": index, "duration_sec": info.duration_sec}
        try:
            deadline = started + args.timeout
            report = worker.analyze(path, f"benchmark-{index}", timeout_sec=args.timeout, deadline=deadline)
            if args.refine:
                middle = info.duration_sec / 2
                refinement = worker.refine(path, f"benchmark-{index}", [{
                    "candidate_id": "timing-probe", "start_sec": max(0, middle - 4),
                    "end_sec": min(info.duration_sec, middle + 4)}], deadline=deadline)
                row["refinement_sec"] = refinement["metrics"]["total_sec"]
                row["refinement_windows"] = sum(len(item["report"]["windows"]) for item in refinement["refinements"])
                (output / f"refinement-{index}.json").write_text(json.dumps(refinement), encoding="utf-8")
            # This tests extraction only; it does not assert any violation.
            evidence = worker.evidence(path, {
                "beginning": [0], "ending": [report["windows"][-1]["frame_timestamps"][-1]]}, deadline=deadline)
            report["benchmark_source_path"] = str(path.resolve())
            (output / f"observations-{index}.json").write_text(json.dumps(report), encoding="utf-8")
            (output / f"extraction-check-{index}.json").write_text(json.dumps(evidence), encoding="utf-8")
            row.update({"complete": True, "windows": len(report["windows"]),
                        "total_with_extraction_sec": round(time.monotonic() - started, 3),
                        **report["metrics"], **observer.metrics()})
        except Exception as exc:
            row.update({"complete": False, "error_type": type(exc).__name__,
                        "elapsed_sec": round(time.monotonic() - started, 3)})
            from app.inference.schema import ModelOutputError
            if isinstance(exc, ModelOutputError):
                row.update({"reason": exc.reason, "attempts": getattr(exc, "attempts", 1),
                            "hit_token_limit": getattr(exc, "hit_token_limit", None)})
        metrics.append(row)
        print(json.dumps(row), flush=True)
    summary = {"model_load_sec": load_sec, "model_revision": observer.model_revision,
               "metrics": metrics}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not all(row["complete"] for row in metrics):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
