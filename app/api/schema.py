from __future__ import annotations

import base64
import binascii
import io
import warnings
from typing import Annotated, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


Text = Annotated[str, Field(min_length=1, max_length=8000, pattern=r"\S")]
Seconds = Annotated[float, Field(ge=0)]


class EvidenceFrame(StrictModel):
    timestamp_sec: Seconds
    image_base64: Annotated[str, Field(min_length=4, max_length=8 * 1024 * 1024)]

    @field_validator("image_base64")
    @classmethod
    def jpeg_only(cls, value):
        try:
            raw = base64.b64decode(value, validate=True)
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as img:
                    if img.format != "JPEG" or img.width * img.height > 20_000_000:
                        raise ValueError("Invalid evidence image")
                    img.verify()
                with Image.open(io.BytesIO(raw)) as img:
                    img.load()
        except (ValueError, OSError, binascii.Error, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as exc:
            raise ValueError("Expected decodable JPEG base64") from exc
        return value


class Violation(StrictModel):
    type: Text
    timecode_sec: Seconds
    description: Text
    evidence_frames: Annotated[list[EvidenceFrame], Field(min_length=1, max_length=3)]


class AnalysisResult(StrictModel):
    """Private, versioned result. Observation reports cannot validate as this type."""
    schema_version: Literal["analysis.v1"]
    analysis_kind: Literal["rules_evaluated", "signs_evaluated"]
    complete: Literal[True]
    coverage_scope: Literal["full_video"]
    rules_reviewed: bool
    source_texts_verified: bool | None = None
    accuracy_status: Literal["unmeasured"] | None = None
    duration_sec: Annotated[float, Field(gt=0)]
    model_id: Text
    model_revision: Text
    rules_version: Text
    pipeline_version: Text
    verdict: Literal["violation", "no_violation"]
    violations: Annotated[list[Violation], Field(max_length=100)]
    rule_ids: Annotated[list[Text], Field(max_length=100)]
    negative_policy_version: Text | None = None

    @model_validator(mode="after")
    def consistent(self):
        if self.analysis_kind == "rules_evaluated":
            if self.rules_reviewed is not True:
                raise ValueError("Reviewed evaluation requires reviewed rules")
        else:
            if (self.rules_reviewed is not False or self.source_texts_verified is not True
                    or self.accuracy_status != "unmeasured"
                    or self.negative_policy_version != "signs-targeted.v1"):
                raise ValueError("Signs evaluation requires explicit source and quality metadata")
        if (self.verdict == "violation") != bool(self.violations):
            raise ValueError("Verdict and events disagree")
        if self.verdict == "no_violation" and self.negative_policy_version is None:
            raise ValueError("No violation requires an explicit negative decision policy")
        if len(self.rule_ids) != len(self.violations):
            raise ValueError("Each event needs its source rule id")
        for event in self.violations:
            if event.timecode_sec > self.duration_sec:
                raise ValueError("Event lies outside video")
            if any(frame.timestamp_sec > self.duration_sec for frame in event.evidence_frames):
                raise ValueError("Evidence lies outside video")
        return self

    def public(self, job_id):
        return {"job_id": job_id, "status": "done", "verdict": self.verdict,
                "violations": [event.model_dump() for event in self.violations]}


class WorkerHello(StrictModel):
    worker_id: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")]
    ready: bool
    pipeline_version: Text


class Attempt(StrictModel):
    worker_id: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")]
    attempt_token: Annotated[str, Field(min_length=32, max_length=128)]


class Completion(Attempt):
    result: AnalysisResult


class Failure(Attempt):
    code: Literal["analysis_failed", "invalid_video", "deadline_exceeded", "worker_unavailable"]
    retryable: bool = False
