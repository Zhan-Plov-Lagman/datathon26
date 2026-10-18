"""Synthetic-only tests against actual fetched B/C modules, without merging branches."""
import base64
import copy
import importlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.inference.worker import LocalGPUWorker, WorkerBusy
from app.inference.assessments import parse_candidates
from app.inference.departure import VERSION, parse_features
from app.inference.schema import ModelOutputError
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


class SyntheticDetectionObserver(SyntheticObserver):
    """Scripted responses ONLY for generated color fixtures; no model accuracy claim."""
    fail = False
    empty = False

    def _observe_once(self, frames, previous_context, deadline, *, parser, **kwargs):
        if self.fail:
            raise ModelOutputError('invalid_schema')
        times = [f.timestamp_sec for f in frames]
        if parser is parse_candidates:
            data = {'operation_context':'Synthetic fixture', 'visibility_limits':[],
                    'observations':[], 'episodes':[]}
            if not self.empty:
                data['observations'] = [{'start_frame':0, 'end_frame':len(frames)-1,
                                         'description':'Synthetic object crosses a test boundary'}]
                data['episodes'] = [{'inside_frame':0, 'anchor_frame':1,
                                     'last_frame':len(frames)-1, 'observation_indices':[0]}]
        elif parser is parse_features:
            data = {'same_person':True, 'cash_from_customer':True, 'cash_handling':True,
                    'visible_crossing':True, 'feet':[[.2,.5]]+[[.8,.5]]*(len(frames)-1),
                    'handover_active':[True]*len(frames)}
        else:
            raise AssertionError('Unexpected synthetic stage')
        return parser(json.dumps(data), times)


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

    def detection_evaluator(self, with_context=True):
        evaluator = importlib.import_module('app.jobs.evaluator')
        self.assertTrue(Path(evaluator.__file__).resolve().is_relative_to(self.snapshot))
        registry = copy.deepcopy(self.engine.load_registry(self.snapshot/'config/rules.yaml'))
        for rule in registry['rules']:
            if rule['rule_id'] == 'cashier_leaves_service':
                rule.update(status='reviewed', validated_on_local_cases=True)
                rule['source']['reviewer'] = 'SYNTHETIC TEST ONLY'
        release = {'validation_reference':'SYNTHETIC TEST ONLY',
                   'registry_sha256':evaluator.registry_digest(registry),
                   'model_id':SyntheticObserver.model_id,
                   'model_revision':SyntheticObserver.model_revision,
                   'detector_versions':{'cashier_leaves_service':VERSION},
                   'policy_version':'detection-policy.v1'}

        def synthetic_context(report):
            return {'schema_version':'local_context.draft.v1', 'view_id':'synthetic',
                    'context_version':'test', 'reviewer':'SYNTHETIC TEST ONLY',
                    'reviewed_at':'synthetic-date', 'basis_ref':'synthetic-fixture',
                    'coordinate_space':'normalized_display_after_rotation',
                    'scope':{'job_id':report['job_id'], 'display_width':96,
                             'display_height':64, 'rotation':0},
                    'workplace_polygon':[[.1,.1],[.5,.1],[.5,.9],[.1,.9]],
                    'adjacent_equipment_included':False, 'max_transition_gap_sec':1,
                    'cash_context_confirmation':{'value':True,'basis_ref':'synthetic'},
                    'functional_role_confirmation':{'value':True,'basis_ref':'synthetic'}}
        return evaluator.DetectionEvaluator(registry=registry, release=release,
                       context_provider=synthetic_context if with_context else None)

    def test_current_evaluator_uses_actual_worker_refinement_context_budget_and_jpeg(self):
        worker = LocalGPUWorker(SyntheticDetectionObserver())
        processor = self.pipeline.IntegratedProcessor(worker, self.detection_evaluator())
        with patch.object(worker, 'analyze', wraps=worker.analyze) as analyze, \
             patch.object(worker, 'assess', wraps=worker.assess) as assess, \
             patch.object(worker, 'evidence', wraps=worker.evidence) as evidence:
            result = processor.process(self.video, 'synthetic-evaluator', 30)
        self.assertEqual(result['verdict'], 'violation')
        self.assertEqual(len(result['violations']), 1)
        self.assertEqual(analyze.call_args.kwargs['deadline'], assess.call_args.kwargs['deadline'])
        self.assertEqual(assess.call_args.kwargs['deadline'], evidence.call_args.kwargs['deadline'])
        self.assertGreater(assess.call_args.kwargs['timeout_sec'], 0)
        self.assertLess(assess.call_args.kwargs['timeout_sec'], 25)
        self.assertEqual(assess.call_args.kwargs['local_context']['scope']['job_id'], 'synthetic-evaluator')
        event = result['violations'][0]
        frame = event['evidence_frames'][0]
        self.assertEqual(frame['timestamp_sec'], event['timecode_sec'])
        self.assertTrue(base64.b64decode(frame['image_base64']).startswith(b'\xff\xd8'))

    def test_current_evaluator_preserves_unknown_and_accepts_executed_empty_search(self):
        for empty in (False, True):
            with self.subTest(empty=empty):
                observer = SyntheticDetectionObserver(); observer.empty = empty
                worker = LocalGPUWorker(observer)
                captured = []
                actual_assess = worker.assess
                def capture(*args, **kwargs):
                    bundle = actual_assess(*args, **kwargs)
                    captured.append(bundle)
                    return bundle
                processor = self.pipeline.IntegratedProcessor(worker, self.detection_evaluator(False))
                with patch.object(worker, 'assess', side_effect=capture), \
                     patch.object(worker, 'evidence', wraps=worker.evidence) as evidence:
                    result = processor.process(self.video, 'synthetic-unknown', 30)
                self.assertEqual(result['verdict'], 'no_violation')
                self.assertEqual(result['violations'], [])
                evidence.assert_not_called()
                self.assertTrue(all(v is None for p in captured[0]['assessments'] for v in p['facts'].values()))
                self.assertEqual(bool(captured[0]['assessments']), not empty)

    def test_current_evaluator_does_not_hide_failure_and_same_worker_recovers(self):
        observer = SyntheticDetectionObserver(); observer.fail = True
        worker = LocalGPUWorker(observer)
        processor = self.pipeline.IntegratedProcessor(worker, self.detection_evaluator())
        with self.assertRaises(ModelOutputError):
            processor.process(self.video, 'synthetic-error', 30)
        observer.fail = False
        self.assertEqual(processor.process(self.video, 'synthetic-recovery', 30)['verdict'], 'violation')
        with worker._lock:
            with self.assertRaises(WorkerBusy):
                processor.process(self.video, 'synthetic-busy', 30)
