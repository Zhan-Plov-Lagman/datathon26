"""Only synthetic test processors. Never select these on the jury service."""
import time


class SyntheticProcessor:
    pipeline_version = "synthetic-test-only"

    def process(self, video_path, job_id, timeout_sec):
        return {"schema_version": "analysis.v1", "analysis_kind": "rules_evaluated",
                "complete": True, "coverage_scope": "full_video", "rules_reviewed": True,
                "duration_sec": 3.0, "model_id": "synthetic-test-only", "model_revision": "test",
                "rules_version": "synthetic-test-only", "pipeline_version": self.pipeline_version,
                "verdict": "no_violation", "violations": [], "rule_ids": []}


class HangingProcessor(SyntheticProcessor):
    def process(self, *args):
        time.sleep(60)


def make_processor():
    return SyntheticProcessor()


def make_hanging_processor():
    return HangingProcessor()
