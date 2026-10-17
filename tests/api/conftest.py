import base64
import io

import pytest
from PIL import Image

from app.api.settings import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(storage=tmp_path / "runtime", api_key="public-test-" * 4, worker_key="private-test-" * 4)


@pytest.fixture
def result():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "green").save(buffer, "JPEG")
    return {"schema_version": "analysis.v1", "analysis_kind": "rules_evaluated",
            "complete": True, "coverage_scope": "full_video", "rules_reviewed": True,
            "duration_sec": 3.0, "model_id": "synthetic", "model_revision": "test",
            "rules_version": "synthetic-test-only", "pipeline_version": "test",
            "verdict": "violation", "rule_ids": ["synthetic-rule"], "violations": [{
                "type": "Synthetic event", "timecode_sec": 1.0,
                "description": "Synthetic fixture; not a legal or bank scene.",
                "evidence_frames": [{"timestamp_sec": 1.0,
                    "image_base64": base64.b64encode(buffer.getvalue()).decode()}]}]}
