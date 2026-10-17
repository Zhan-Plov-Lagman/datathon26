"""C's adapter to A's analyzer and B's reviewed evaluator (see WORKER_TRANSPORT.md)."""
from __future__ import annotations

import importlib
import os
import time
from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.api.schema import AnalysisResult, Seconds, StrictModel, Text


class SupportedEvent(StrictModel):
    rule_id: Text
    type: Text
    anchor_sec: Seconds
    description: Text
    evidence_timestamps: Annotated[list[Seconds], Field(min_length=1, max_length=3)]


class RuleDecision(StrictModel):
    complete: Literal[True]
    rules_reviewed: Literal[True]
    rules_version: Text
    supported: Annotated[list[SupportedEvent], Field(max_length=100)]
    negative_policy_version: Text | None = None

    @model_validator(mode="after")
    def justified_negative(self):
        if not self.supported and self.negative_policy_version is None:
            raise ValueError("Empty events require a reviewed negative decision policy")
        return self


def import_factory(spec):
    module, separator, name = spec.partition(":")
    if not separator:
        raise ValueError("Factory must have module:function format")
    return getattr(importlib.import_module(module), name)


class IntegratedProcessor:
    pipeline_version = "coordinator.v1"

    def __init__(self, worker, rules):
        if getattr(rules, "reviewed", False) is not True:
            raise RuntimeError("Reviewed rules are not ready")
        self.worker, self.rules = worker, rules

    def process(self, video_path, job_id, timeout_sec):
        deadline = time.monotonic() + timeout_sec
        report = self.worker.analyze(video_path, job_id, timeout_sec=timeout_sec, deadline=deadline)
        if (report.get("complete") is not True or report.get("coverage_scope") != "full_video"
                or report.get("schema_version") != "observations.v1"):
            raise ValueError("Incomplete observations")
        # B performs applicability, exceptions, deduplication and refinement via worker.refine.
        decision = RuleDecision.model_validate(self.rules.evaluate(
            report, video_path=video_path, worker=self.worker, deadline=deadline))
        requests = {str(index): event.evidence_timestamps for index, event in enumerate(decision.supported)}
        evidence = self.worker.evidence(video_path, requests, deadline=deadline) if requests else {}
        events = []
        for index, event in enumerate(decision.supported):
            if time.monotonic() >= deadline:
                raise TimeoutError
            frames = evidence[str(index)]
            events.append({"type": event.type, "timecode_sec": event.anchor_sec,
                           "description": event.description, "evidence_frames": frames})
        if time.monotonic() >= deadline:
            raise TimeoutError
        return AnalysisResult.model_validate({
            "schema_version": "analysis.v1", "analysis_kind": "rules_evaluated",
            "complete": True, "coverage_scope": "full_video", "rules_reviewed": True,
            "duration_sec": report["video"]["duration_sec"],
            "model_id": report["model"]["id"], "model_revision": report["model"]["revision"],
            "rules_version": decision.rules_version, "pipeline_version": self.pipeline_version,
            "negative_policy_version": decision.negative_policy_version,
            "verdict": "violation" if events else "no_violation", "violations": events,
            "rule_ids": [event.rule_id for event in decision.supported],
        }).model_dump()


def create_processor():
    spec = os.environ.get("FORTE_RULES_FACTORY")
    if not spec:
        raise RuntimeError("FORTE_RULES_FACTORY is required; observations alone are not a verdict")
    rules = import_factory(spec)()
    if getattr(rules, "reviewed", False) is not True:
        raise RuntimeError("Rules registry is not reviewed")
    # Heavy dependencies are imported only on the GPU machine, after the rules check.
    from app.inference.qwen import QwenObserver
    from app.inference.worker import LocalGPUWorker

    observer = QwenObserver(os.environ.get("FORTE_MODEL_DIR", "models/Qwen3-VL-2B-Instruct-shards"))
    return IntegratedProcessor(LocalGPUWorker(observer), rules)
