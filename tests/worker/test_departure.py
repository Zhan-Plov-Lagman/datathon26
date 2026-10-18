import copy
import importlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from app.inference.assessments import build_bundle, extract_assessments, parse_candidates
from app.inference.departure import ground_facts, inside, parse_features, validate_context
from app.inference.schema import ModelOutputError
from app.video.sampling import SampledFrame, VideoInfo
from test_assessments import report, response


def features():
    return {"same_person": True, "cash_from_customer": True, "cash_handling": True,
            "visible_crossing": True, "feet": [[.2,.5],[.8,.5],[.8,.5]],
            "handover_active": [True,True,True]}


def context():
    return {"schema_version":"local_context.draft.v1", "view_id":"synthetic-view",
            "context_version":"test", "reviewer":"synthetic-reviewer", "reviewed_at":"synthetic-date",
            "basis_ref":"synthetic-only", "coordinate_space":"normalized_display_after_rotation",
            "scope":{"job_id":"synthetic","display_width":64,"display_height":64,"rotation":0},
            "workplace_polygon":[[.1,.1],[.5,.1],[.5,.9],[.1,.9]],
            "adjacent_equipment_included":False, "max_transition_gap_sec":1.1,
            "cash_context_confirmation":{"value":True,"basis_ref":"synthetic-cash"},
            "functional_role_confirmation":{"value":True,"basis_ref":"synthetic-role"}}


class DepartureTests(unittest.TestCase):
    def test_parser_rejects_bad_geometry_types_and_missing_frames(self):
        for change in (lambda d:d.update(same_person="true"),
                       lambda d:d.update(feet=[[2,0]]*3),
                       lambda d:d.update(handover_active=[True])):
            data=features();change(data)
            with self.assertRaises(ModelOutputError):parse_features(json.dumps(data),[0,1,2])

    def test_only_all_unknown_arrays_can_be_resized_without_fabrication(self):
        data=features();data['feet']=[None]*4
        parsed=parse_features(json.dumps(data),[0,1,2])
        self.assertEqual(parsed['feet'],[None]*3)
        self.assertEqual(parsed['_normalizations'],['feet_all_unknown_length'])
        data['feet']=[None,None,None,[.2,.3]]
        with self.assertRaises(ModelOutputError):parse_features(json.dumps(data),[0,1,2])

    def test_crossing_and_unfinished_handover_generate_grounded_bools(self):
        c=validate_context(context(),report())
        episode=parse_candidates(json.dumps(response()),[0,1,2])["episodes"][0]
        facts,sources=ground_facts(features(),episode,report()["windows"][0],c,
                                  [{"window_id":1,"observation_index":0}])
        self.assertTrue(facts["cashier_left_workplace"])
        self.assertTrue(facts["service_in_progress"])
        self.assertFalse(facts["service_completed"])
        self.assertEqual(sources["service_completed"]["origin"],"mixed")
        self.assertTrue(sources["service_completed"]["context_ref"])

    def test_absent_context_occlusion_boundary_gap_and_completed_transfer(self):
        episode=parse_candidates(json.dumps(response()),[0,1,2])["episodes"][0]
        def run(f,c):return ground_facts(f,episode,report()["windows"][0],c,[])[0]
        self.assertTrue(all(v is None for v in run(features(),None).values()))
        f=features();f["feet"][1]=None
        self.assertIsNone(run(f,context())["cashier_left_workplace"])
        f=features();f["feet"][1]=[.5,.5]
        self.assertIsNone(run(f,context())["cashier_left_workplace"])
        c=context();c["max_transition_gap_sec"]=.1
        self.assertIsNone(run(features(),c)["service_in_progress"])
        f=features();f["handover_active"]=[False]*3
        self.assertIsNone(run(f,context())["service_completed"])
        self.assertIsNone(inside([.5,.5],context()["workplace_polygon"]))

    def test_context_binding_rejects_other_job_view_and_invented_review(self):
        for change in (lambda c:c["scope"].update(job_id="other"),
                       lambda c:c["scope"].update(display_width=128),
                       lambda c:c.update(reviewer="")):
            c=context();change(c)
            with self.assertRaises(ValueError):validate_context(c,report())

    def test_receipt_only_after_both_stages_complete_including_empty_search(self):
        parsed=parse_candidates(json.dumps(response()),[0,1,2]);parsed["episodes"]=[]
        observer=Mock();observer._observe_once.side_effect=[parsed,features()]
        with patch("app.inference.assessments.probe_video",return_value=VideoInfo(3,0,64,64,0)), \
             patch("app.inference.assessments.sample_frames",return_value={i:SampledFrame(i,None) for i in range(3)}):
            bundle=extract_assessments(observer,"synthetic.mov",report(), refine_candidates=False)
            run=bundle["diagnostics"]["search"]["detector_runs"][0]
            self.assertEqual(run["processed_window_ids"],[0,1])
            self.assertEqual(run["refinement_status"],"not_needed")
            self.assertEqual(bundle["assessments"],[])
            observer._observe_once.side_effect=[parsed,ModelOutputError("invalid_schema"),ModelOutputError("invalid_schema")]
            with self.assertRaises(ModelOutputError):extract_assessments(observer,"synthetic.mov",report(), refine_candidates=False)
        self.assertEqual(build_bundle(report(),[parsed])["diagnostics"]["search"]["detector_runs"],[])
        with self.assertRaises(ValueError):build_bundle(report(),[parsed],executed=[])

    def test_schema_retry_regenerates_once_without_changing_deadline_or_values(self):
        parsed=parse_candidates(json.dumps(response()),[0,1,2]);parsed['episodes']=[]
        observer=Mock();observer._observe_once.side_effect=[ModelOutputError('invalid_schema'),parsed,features()]
        with patch('app.inference.assessments.probe_video',return_value=VideoInfo(3,0,64,64,0)), \
             patch('app.inference.assessments.sample_frames',return_value={i:SampledFrame(i,None) for i in range(3)}):
            bundle=extract_assessments(observer,'synthetic.mov',report(), refine_candidates=False)
        self.assertEqual(bundle['metrics']['format_retries'],1)
        calls=observer._observe_once.call_args_list
        self.assertTrue(calls[1].kwargs['concise'])
        self.assertEqual(calls[0].args[2],calls[1].args[2])

    @unittest.skipUnless(os.environ.get("FORTE_PEER_SNAPSHOT"),"Set B/C snapshot")
    def test_actual_c_and_b_accept_producer_receipt_with_synthetic_registry(self):
        import app
        root=Path(os.environ["FORTE_PEER_SNAPSHOT"]).resolve()
        app.__path__.append(str(root/"app"))
        detection=importlib.import_module("app.jobs.detection")
        rules=importlib.import_module("app.rules")
        registry=rules.load_registry(root/"config/rules.yaml")
        # Synthetic policy test ONLY; no on-disk registry is modified.
        registry=copy.deepcopy(registry)
        for rule in registry["rules"]:
            if rule["rule_id"]=="cashier_leaves_service":
                rule["status"]="reviewed";rule["validated_on_local_cases"]=True
                rule["source"]["reviewer"]="synthetic-policy-test"
        original=report();original["model"]={"id":"synthetic"}
        parsed=parse_candidates(json.dumps(response()),[0,1,2])
        observer=Mock();observer._observe_once.side_effect=[parsed,features()]
        with patch("app.inference.assessments.probe_video",return_value=VideoInfo(3,0,64,64,0)), \
             patch("app.inference.assessments.sample_frames",return_value={i:SampledFrame(i,None) for i in range(3)}):
            bundle=extract_assessments(observer,"synthetic.mov",original, refine_candidates=False)
        result=detection.finalize_bundle(original,bundle,supported_rule_ids=["cashier_leaves_service"],registry=registry)
        self.assertEqual(result["verdict"],"no_violation")
        self.assertTrue(all(v is None for v in bundle["assessments"][0]["facts"].values()))
        bundle["diagnostics"]["search"]["detector_runs"][0]["processed_window_ids"]=[0]
        with self.assertRaises(ValueError):
            detection.finalize_bundle(original,bundle,supported_rule_ids=["cashier_leaves_service"],registry=registry)


if __name__=="__main__":unittest.main()
