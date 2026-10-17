"""Local benchmark; stdout contains technical metrics, never video descriptions."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from app.inference.worker import LocalGPUWorker
from app.inference.qwen import QwenObserver
from app.video.sampling import probe_video


def run_stability(worker, video: Path, output: Path, timeout: float) -> list[dict]:
    """One loaded worker, real development input, synthetic failures, safe metrics."""
    from app.video.sampling import DeadlineExceeded, VideoError

    invalid = output / "synthetic-invalid.mov"
    invalid.write_bytes(b"synthetic invalid video")
    torch = worker.analyzer.observer.torch
    cases = [("valid", video, None), ("invalid_input", invalid, VideoError),
             ("valid_after_invalid", video, None),
             ("expired_deadline", video, DeadlineExceeded),
             ("short_deadline", video, DeadlineExceeded),
             ("valid_after_deadlines", video, None)]
    rows = []
    for index, (kind, path, expected) in enumerate(cases):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = worker.analyzer.observer.metrics().get("format_retries", 0)
        started = time.monotonic()
        deadline = started - 1 if kind == "expired_deadline" else started + (3 if kind == "short_deadline" else timeout)
        row = {"sample_index": index, "case": kind, "expected_error": expected.__name__ if expected else None}
        try:
            report = worker.analyze(path, f"stability-{index}", timeout, deadline=deadline)
            (output / f"observations-{index}.json").write_text(json.dumps(report), encoding="utf-8")
            row.update({"complete": True, "expected_outcome": expected is None,
                        "decode_sec": report["metrics"]["decode_sec"],
                        "analysis_sec": report["metrics"]["total_sec"], "windows": len(report["windows"])})
            del report
        except Exception as exc:
            row.update({"complete": False, "error_type": type(exc).__name__,
                        "expected_outcome": expected is not None and isinstance(exc, expected)})
        # Collect after the except block so exception tracebacks cannot retain tensors.
        gc.collect()
        torch.cuda.synchronize()
        stats = worker.analyzer.observer.metrics()
        row.update({"elapsed_sec": round(time.monotonic() - started, 3), **stats,
                    "format_retries_this_task": stats.get("format_retries", 0) - before,
                    "idle_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
                    "idle_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1)})
        rows.append(row)
        (output / "stability-metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video-dir")
    inputs.add_argument("--video", help="Benchmark one explicit local MOV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", default="models/Qwen3-VL-2B-Instruct-shards")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--refine", action="store_true", help="Also test a middle interval, not a labeled event")
    parser.add_argument("--stability", action="store_true", help="Six sequential cases on one explicit development video")
    args = parser.parse_args()
    if args.stability and (not args.video or args.refine):
        parser.error("--stability requires --video and cannot be combined with --refine")
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
    if args.stability:
        rows = run_stability(worker, paths[0], output, args.timeout)
        (output / "metrics.json").write_text(json.dumps({
            "model_load_sec": load_sec, "model_revision": observer.model_revision,
            "metrics": rows}, indent=2), encoding="utf-8")
        if not all(row["expected_outcome"] for row in rows):
            raise SystemExit(1)
        return
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
