import copy
import json
import importlib.util
import os
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

from app.inference.assessments import FACTS, build_bundle, extract_assessments, parse_candidates
from app.inference.schema import ModelOutputError
from app.inference.qwen import QwenObserver
from app.inference.worker import LocalGPUWorker, WorkerBusy
from app.video.sampling import DeadlineExceeded, SampledFrame, VideoInfo


def response():
    return {"operation_context": "Synthetic movement", "visibility_limits": [],
            "observations": [{"start_frame": 0, "end_frame": 2, "description": "Synthetic motion"}],
            "episodes": [{"inside_frame": 0, "anchor_frame": 1, "last_frame": 2,
                          "observation_indices": [0], "claims": dict.fromkeys(FACTS)}]}


def report():
    return {"schema_version": "observations.v1", "analysis_kind": "observations_only",
            "complete": True, "coverage_scope": "full_video", "job_id": "synthetic",
            "video": {"duration_sec": 3, "width": 64, "height": 64, "rotation": 0, "start_time_sec": 0},
            "windows": [{"window_id": 0, "window_start_sec": 0, "window_end_sec": 3,
                         "frame_timestamps": [0, 1, 2], "operation_context": "Synthetic",
                         "visibility_limits": [], "observations": []}]}


class AssessmentTests(unittest.TestCase):
    def test_temporal_refs_include_observation_covering_anchor(self):
        data = response()
        data["observations"] = [{"start_frame": 0, "end_frame": 0, "description": "Synthetic before"},
                                {"start_frame": 1, "end_frame": 2, "description": "Synthetic after"}]
        parsed = parse_candidates(json.dumps(data), [0, 1, 2])
        self.assertEqual(parsed["episodes"][0]["observation_indices"], [0, 1])
        self.assertTrue(all(v is None for v in parsed["episodes"][0]["claims"].values()))

    def test_single_visibility_string_is_preserved_as_limitation(self):
        data = response(); data["visibility_limits"] = "Synthetic occlusion"
        parsed = parse_candidates(json.dumps(data), [0, 1, 2])
        self.assertEqual(parsed["visibility_limits"], ["Synthetic occlusion"])

    def test_custom_prompt_uses_indices_without_seconds(self):
        observer = object.__new__(QwenObserver)
        observer.processor = Mock(side_effect=RuntimeError("synthetic processor stop"))
        frames = [SampledFrame(7.25, None), SampledFrame(9.5, None)]
        with self.assertRaises(RuntimeError):
            observer._observe_once(frames, "", None, system_prompt="Synthetic instructions")
        messages = observer.processor.apply_chat_template.call_args.args[0]
        text = " ".join(x.get("text", "") for x in messages[1]["content"])
        self.assertIn("Allowed frame indices: 0, 1", text)
        self.assertNotIn("7.250000", text)
        self.assertNotIn("9.500000", text)

    def test_minimal_visual_response_does_not_invent_fact_claims(self):
        data = response()
        del data["episodes"][0]["claims"]
        parsed = parse_candidates(json.dumps(data), [0, 1, 2])
        self.assertEqual(parsed["episodes"][0]["claims"], dict.fromkeys(FACTS))

    @unittest.skipUnless(os.environ.get("FORTE_PEER_SNAPSHOT"), "Set local B snapshot")
    def test_actual_b_accepts_bundle_and_keeps_candidate(self):
        root = Path(os.environ["FORTE_PEER_SNAPSHOT"])
        spec = importlib.util.spec_from_file_location("peer_rules", root / "app/rules/engine.py")
        engine = importlib.util.module_from_spec(spec); spec.loader.exec_module(engine)
        parsed = parse_candidates(json.dumps(response()), [0, 1, 2])
        bundle = build_bundle(report(), [parsed])
        result = engine.evaluate(bundle["report"], bundle["assessments"],
                                 engine.load_registry(root / "config/rules.yaml"))
        self.assertEqual(result["supported"], [])
        self.assertEqual(result["candidates"][0]["decision"], "candidate")
        self.assertFalse(result["public_verdict_ready"])

    def test_grounded_refs_new_ids_and_conflicts_remain_unknown(self):
        data = response()
        data["episodes"][0]["claims"] = dict.fromkeys(FACTS, True)
        original = report()
        before = copy.deepcopy(original)
        bundle = build_bundle(original, [parse_candidates(json.dumps(data), [0, 1, 2])])
        self.assertEqual(original, before)
        self.assertEqual(bundle["assessments"][0]["anchor_sec"], 1)
        self.assertEqual(bundle["assessments"][0]["observation_refs"][0]["window_id"], 1)
        self.assertTrue(bundle["diagnostics"]["episodes"][0]["service_conflict"])
        self.assertTrue(all(v is None for v in bundle["assessments"][0]["facts"].values()))
        self.assertFalse(bundle["assessments"][0]["refinement_complete"])

    def test_bad_indices_string_bool_and_ungrounded_anchor_rejected(self):
        for change in (lambda e: e.update(anchor_frame=99),
                       lambda e: e["claims"].update(service_completed="false"),
                       lambda e: e.update(observation_indices=[7])):
            data = response(); change(data["episodes"][0])
            with self.assertRaises(ModelOutputError):
                parse_candidates(json.dumps(data), [0, 1, 2])

    def test_overlap_not_invented_identity_and_empty_not_negative(self):
        source = report(); source["windows"].append({**source["windows"][0], "window_id": 1})
        parsed = parse_candidates(json.dumps(response()), [0, 1, 2])
        bundle = build_bundle(source, [parsed, parsed])
        self.assertEqual(bundle["assessments"], [])
        self.assertTrue(all(d["overlap_ambiguous"] for d in bundle["diagnostics"]["episodes"]))
        self.assertFalse(bundle["public_verdict_ready"])
        self.assertNotIn("verdict", bundle)

    def test_deadline_retry_pts_and_context(self):
        observer = Mock()
        parsed = parse_candidates(json.dumps(response()), [0, 1, 2])
        features = {"same_person": None, "cash_from_customer": None, "cash_handling": None,
                    "visible_crossing": None, "feet": [None]*3, "handover_active": [None]*3}
        observer._observe_once.side_effect = [ModelOutputError("invalid_json"), parsed, features]
        deadline = time.monotonic() + 20
        with patch("app.inference.assessments.probe_video", return_value=VideoInfo(3, 0, 64, 64, 0)), \
             patch("app.inference.assessments.sample_frames", return_value={i: SampledFrame(i, None) for i in range(3)}):
            bundle = extract_assessments(observer, "synthetic.mov", report(), deadline=deadline)
        self.assertEqual(len(bundle["assessments"]), 1)
        self.assertEqual(observer._observe_once.call_args.args[2], deadline)
        self.assertEqual(observer._observe_once.call_count, 3)
        self.assertEqual(bundle["metrics"]["format_retries"], 1)
        with patch("app.inference.assessments.probe_video", return_value=VideoInfo(3, 0, 64, 64, 0)), \
             patch("app.inference.assessments.sample_frames", return_value={i: SampledFrame(i+0.01, None) for i in range(3)}):
            with self.assertRaises(ValueError):
                extract_assessments(observer, "synthetic.mov", report())
        with self.assertRaises(DeadlineExceeded):
            extract_assessments(observer, "synthetic.mov", report(), deadline=time.monotonic()-1)
        with self.assertRaises(ValueError):
            extract_assessments(observer, "synthetic.mov", report(), local_context={})

    def test_worker_lock_released_after_failure(self):
        worker = LocalGPUWorker(Mock())
        with self.assertRaises(ValueError):
            worker.assess("synthetic.mov", {}, local_context={})
        self.assertTrue(worker._lock.acquire(False))
        try:
            with self.assertRaises(WorkerBusy):
                worker.assess("synthetic.mov", report())
        finally:
            worker._lock.release()


if __name__ == "__main__":
    unittest.main()
