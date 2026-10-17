"""Synthetic-only tests against actual fetched B/C modules, without merging branches."""
import base64
import importlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.inference.worker import LocalGPUWorker, WorkerBusy
from app.video.sampling import VideoError


class SyntheticObserver:
    model_id = "synthetic-test-only"
    model_revision = "synthetic"

    def observe(self, frames, previous_context, deadline):
        return {"operation_context": "synthetic colors", "visibility_limits": [],
                "observations": [{"start_frame": 0, "end_frame": len(frames) - 1,
                                  "start_sec": frames[0].timestamp_sec,
                                  "end_sec": frames[-1].timestamp_sec,
                                  "description": "Synthetic color pattern"}]}


@unittest.skipUnless(os.environ.get("FORTE_PEER_SNAPSHOT"), "Set FORTE_PEER_SNAPSHOT to prepared peer snapshot")
class PeerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app
        cls.snapshot = Path(os.environ["FORTE_PEER_SNAPSHOT"]).resolve()
        app.__path__.append(str(cls.snapshot / "app"))
        cls.engine = importlib.import_module("app.rules.engine")
        cls.pipeline = importlib.import_module("app.jobs.pipeline")
        cls.schema = importlib.import_module("app.api.schema")
        # Ensure this test exercised the requested snapshot, not an unrelated checkout.
        for module in (cls.engine, cls.pipeline, cls.schema):
            if not Path(module.__file__).resolve().is_relative_to(cls.snapshot):
                raise RuntimeError("Peer module was loaded from a different checkout")
        cls.temp = tempfile.TemporaryDirectory()
        cls.video = Path(cls.temp.name) / "synthetic.mov"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=96x64:rate=10:duration=2", "-c:v", "libx264", str(cls.video)],
                       check=True, capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.worker = LocalGPUWorker(SyntheticObserver())

    def packet(self, report, rule):
        return {"rule_id": rule["rule_id"], "event_id": "synthetic-episode",
                "observation_refs": [{"window_id": 0, "observation_index": 0}],
                "anchor_sec": report["windows"][0]["frame_timestamps"][0],
                "facts": {**{key: True for key in rule["applicability"] + rule["required_facts"]},
                          **{key: False for key in rule["exceptions"]}},
                "refinement_complete": True, "full_sequence_visible": True}

    def test_actual_b_accepts_a_report_and_blocks_hypothesis(self):
        report = self.worker.analyze(self.video, "synthetic")
        self.engine.validate_report(report)
        registry = self.engine.load_registry(self.snapshot / "config/rules.yaml")
        rule = next(rule for rule in registry["rules"] if rule["status"] == "hypothesis")
        result = self.engine.evaluate(report, [self.packet(report, rule)], registry)
        self.assertEqual(result["supported"], [])
        self.assertIn("rule_not_reviewed", result["candidates"][0]["reasons"])
        self.assertFalse(result["public_verdict_ready"])

    def test_current_b_result_is_not_c_rule_decision(self):
        from pydantic import ValidationError
        report = self.worker.analyze(self.video, "synthetic")
        result = self.engine.evaluate(report, [], self.engine.load_registry(self.snapshot / "config/rules.yaml"))
        with self.assertRaises(ValidationError):
            self.pipeline.RuleDecision.model_validate(result)

    def synthetic_rules(self):
        owner = self
        # This registry and adapter exist ONLY in tests, never as a production factory.
        rule = {"rule_id": "synthetic", "type": "synthetic_color", "title": "Synthetic only",
                "explanation": "Synthetic test event", "status": "reviewed", "validated_on_local_cases": True,
                "applicability": ["synthetic_input"], "required_facts": ["synthetic_color"], "exceptions": [],
                "source": {"document": "Synthetic fixture, not law", "clause": "test", "url": "https://example.invalid/test",
                           "edition_date": "synthetic", "checked_on": "synthetic", "reviewer": "test-only"}}
        registry = {"schema_version": "rules.v1", "version": "synthetic-only", "rules": [rule]}

        class Rules:
            reviewed = True

            def evaluate(self, report, *, video_path, worker, deadline):
                refined = worker.refine(video_path, report["job_id"], [
                    {"candidate_id": "synthetic", "start_sec": 0.0, "end_sec": 1.0}], deadline=deadline)
                detail = refined["refinements"][0]["report"]
                result = owner.engine.evaluate(detail, [owner.packet(detail, rule)], registry)
                return {"complete": True, "rules_reviewed": True, "rules_version": registry["version"],
                        "supported": [{"rule_id": item["rule_id"], "type": item["type"],
                                       "anchor_sec": item["anchor_sec"], "description": item["description"],
                                       "evidence_timestamps": [item["anchor_sec"]]} for item in result["supported"]]}

        return Rules()

    def test_c_adapter_consumes_a_refinement_and_real_jpeg_with_synthetic_rules(self):
        processor = self.pipeline.IntegratedProcessor(self.worker, self.synthetic_rules())
        with patch.object(self.worker, "refine", wraps=self.worker.refine) as refine:
            result = processor.process(self.video, "synthetic", 30)
        self.assertEqual(refine.call_count, 1)
        public = self.schema.AnalysisResult.model_validate(result).public("synthetic")
        self.assertEqual(public["verdict"], "violation")
        jpeg = base64.b64decode(public["violations"][0]["evidence_frames"][0]["image_base64"])
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))

    def test_c_propagates_a_failure_and_worker_recovers(self):
        processor = self.pipeline.IntegratedProcessor(self.worker, self.synthetic_rules())
        invalid = Path(self.temp.name) / "invalid.mov"
        invalid.write_bytes(b"synthetic invalid input")
        with self.assertRaises(VideoError):
            processor.process(invalid, "synthetic-bad", 30)
        self.assertEqual(processor.process(self.video, "synthetic-after", 30)["verdict"], "violation")

    def test_c_rejects_unreviewed_rules_and_busy_worker(self):
        with self.assertRaises(RuntimeError):
            self.pipeline.IntegratedProcessor(self.worker, object())
        processor = self.pipeline.IntegratedProcessor(self.worker, self.synthetic_rules())
        with patch.object(self.worker, "analyze", side_effect=WorkerBusy()):
            with self.assertRaises(WorkerBusy):
                processor.process(self.video, "synthetic-busy", 30)
