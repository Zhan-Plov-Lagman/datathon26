"""Local human-review drafts using the existing observation pipeline."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path


def timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    minutes, remainder = divmod(milliseconds, 60000)
    return f"{minutes:02d}:{remainder / 1000:06.3f}"


def literal(value: str) -> str:
    # Model text stays literal in Markdown: no embedded links, HTML or images.
    value = " ".join(value.split())
    for char in "\\`*_{}[]<>()#!|":
        value = value.replace(char, "\\" + char)
    return value


def render_draft(report: dict, video_name: str) -> str:
    if report.get("complete") is not True or report.get("schema_version") != "observations.v1":
        raise ValueError("A completed observation report is required")
    lines = ["# Черновик для проверки человеком", "",
             f"Файл: {literal(video_name)}", "Кто просмотрел: __________", "",
             "**НЕ ПРОВЕРЕНО ЧЕЛОВЕКОМ. Это предположения модели, не разметка и не вердикт.**",
             "Модель видит отдельные кадры: возможны пропуски, ошибки и повторы между окнами.",
             "Времена ниже — времена выбранных кадров, а не точные границы действия.",
             "Сначала посмотри видео, затем исправь черновик и добавь пропущенное.", ""]
    for window in report["windows"]:
        lines += [f"## Окно {window['window_id']}: "
                  f"{timestamp(window['window_start_sec'])}–{timestamp(window['window_end_sec'])}", "",
                  "Предположение модели о ситуации: " + literal(window["operation_context"]), "",
                  "Действия по версии модели:", ""]
        for observation in window["observations"]:
            lines.append(f"- {timestamp(observation['start_sec'])}–{timestamp(observation['end_sec'])} — "
                         + literal(observation["description"]))
        if not window["observations"]:
            lines.append("- Модель не выделила действий. Это не доказывает их отсутствие.")
        lines += ["", "Ограничения видимости по версии модели:", ""]
        lines += ["- " + literal(item) for item in window["visibility_limits"]]
        if not window["visibility_limits"]:
            lines.append("- Модель не указала ограничений; проверь обзор самостоятельно.")
        lines += ["", "Мои исправления / пропущенные действия с временем: __________",
                  "Статус окна: не проверено / исправлено / подтверждено / не разобрал", ""]
    lines += ["## После просмотра", "", "Просмотрел видео целиком: нет / да",
              "Что осталось неясным: __________", "Вопросы участнику B: __________", "",
              "Хранить только локально в контуре команды. Не отправлять в облачный чат или Git.", ""]
    return "\n".join(lines)


def create_draft(analyzer, video: Path, output: Path, index: int, timeout: float) -> bool:
    """One job; never expose scene text or exception messages to stdout."""
    started = time.monotonic()
    try:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            report = analyzer.analyze(video, f"review-{index}", timeout)
        draft = render_draft(report, video.name)
        report["review_metadata"] = {"human_reviewed": False, "language": "ru",
                                     "source_path": str(video.resolve())}
        # The enclosing run directory is exclusive; never overwrite human edits.
        (output / f"{index:03d}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / f"{index:03d}.md").write_text(draft, encoding="utf-8")
        status = {"index": index, "complete": True, "windows": len(report["windows"]),
                  "elapsed_sec": round(time.monotonic() - started, 3)}
    except Exception as exc:
        status = {"index": index, "complete": False, "error_type": type(exc).__name__,
                  "elapsed_sec": round(time.monotonic() - started, 3)}
        (output / f"{index:03d}.failed.json").write_text(json.dumps(status), encoding="utf-8")
    print(json.dumps(status), flush=True)
    return status["complete"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Russian drafts for human review; no verdict")
    parser.add_argument("--video", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-dir", default="models/Qwen3-VL-2B-Instruct-shards")
    parser.add_argument("--timeout-sec", type=float, default=240)
    args = parser.parse_args()
    if not math.isfinite(args.timeout_sec) or args.timeout_sec <= 0:
        parser.error("Timeout must be finite and positive")
    if not all(video.is_file() for video in args.video):
        parser.error("An input video does not exist")
    output = args.output_dir.resolve()
    if not output.is_relative_to(Path("artifacts").resolve()):
        parser.error("Keep private drafts inside the ignored artifacts directory")
    if output.exists():
        parser.error("Output directory exists; choose a new run directory")
    output.mkdir(parents=True)
    # Third-party logs remain local too; only safe metrics reach the terminal.
    with (output / "runtime.log").open("w", encoding="utf-8") as log:
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                from app.inference.qwen import QwenObserver
                from app.inference.pipeline import LocalVideoAnalyzer
                observer = QwenObserver(args.model_dir, language="ru")
                analyzer = LocalVideoAnalyzer(observer)
        except Exception as exc:
            print(json.dumps({"complete": False, "stage": "load", "error_type": type(exc).__name__}))
            return 1
        outcomes = []
        for index, video in enumerate(args.video, 1):
            with contextlib.redirect_stderr(log):
                outcomes.append(create_draft(analyzer, video, output, index, args.timeout_sec))
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
