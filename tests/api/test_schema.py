import base64
import io

import pytest
from PIL import Image
from pydantic import ValidationError

from app.api.schema import AnalysisResult


@pytest.mark.parametrize("field,value", [("verdict", "no_violation"), ("complete", False),
    ("coverage_scope", "interval"), ("rules_reviewed", False), ("duration_sec", float("inf")),
    ("duration_sec", 0), ("analysis_kind", "observations_only"), ("rule_ids", [])])
def test_invalid_results_rejected(result, field, value):
    result[field] = value
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(result)


@pytest.mark.parametrize("value", [-1, 4, float("nan")])
def test_event_timestamps_checked(result, value):
    result["violations"][0]["timecode_sec"] = value
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(result)


def test_frame_timestamps_checked(result):
    result["violations"][0]["evidence_frames"][0]["timestamp_sec"] = 4.0
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(result)


@pytest.mark.parametrize("value", ["not-base64", "data:image/jpeg;base64,abcd", base64.b64encode(b"not a jpeg").decode()])
def test_malformed_jpeg(result, value):
    result["violations"][0]["evidence_frames"][0]["image_base64"] = value
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(result)


def test_png_cannot_masquerade_as_jpeg(result):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, "PNG")
    result["violations"][0]["evidence_frames"][0]["image_base64"] = base64.b64encode(buffer.getvalue()).decode()
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(result)


def test_jpeg_really_decodes(result):
    parsed = AnalysisResult.model_validate(result)
    frame = parsed.violations[0].evidence_frames[0]
    with Image.open(io.BytesIO(base64.b64decode(frame.image_base64))) as image:
        image.load()
        assert image.size == (16, 16)
