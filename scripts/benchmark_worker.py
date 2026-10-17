"""Local benchmark; stdout contains technical metrics, never video descriptions."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.evidence.frames import extract_evidence
from app.inference.pipeline import LocalVideoAnalyzer
from app.inference.qwen import QwenObserver
from app.video.sampling import probe_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", default="models/Qwen3-VL-2B-Instruct-shards")
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    # Select by duration only, without reading labels or inferring from filenames.
    paths = list(Path(args.video_dir).glob("*.mov"))
    if not paths:
        raise ValueError("No MOV inputs found")
    ordered = sorted(((p, probe_video(p)) for p in paths), key=lambda item: item[1].duration_sec)
    selected = [ordered[0]] if len(ordered) == 1 else [ordered[0], ordered[-1]]
    started = time.monotonic()
    observer = QwenObserver(args.model_dir)
    load_sec = round(time.monotonic() - started, 3)
    analyzer = LocalVideoAnalyzer(observer)
    metrics = []
    for index, (path, info) in enumerate(selected):
        started = time.monotonic()
        row = {"sample_index": index, "duration_sec": info.duration_sec}
        try:
            report = analyzer.analyze(path, f"benchmark-{index}", timeout_sec=args.timeout)
            # This tests extraction only; it does not assert any violation.
            evidence = extract_evidence(path, [report["windows"][-1]["frame_timestamps"][-1]],
                                        deadline=started + args.timeout)
            report["benchmark_source_path"] = str(path.resolve())
            (output / f"observations-{index}.json").write_text(json.dumps(report), encoding="utf-8")
            (output / f"extraction-check-{index}.json").write_text(json.dumps(evidence), encoding="utf-8")
            row.update({"complete": True, "windows": len(report["windows"]),
                        "total_with_extraction_sec": round(time.monotonic() - started, 3),
                        **report["metrics"]})
        except Exception as exc:
            row.update({"complete": False, "error_type": type(exc).__name__,
                        "elapsed_sec": round(time.monotonic() - started, 3)})
        metrics.append(row)
        print(json.dumps(row), flush=True)
    summary = {"model_load_sec": load_sec, "model_revision": observer.model_revision,
               "metrics": metrics}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not all(row["complete"] for row in metrics):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
