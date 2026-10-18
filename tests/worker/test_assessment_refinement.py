import copy
import importlib
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

from app.inference.assessments import extract_assessments, parse_candidates
from app.inference.schema import ModelOutputError
from app.video.sampling import SampledFrame, VideoInfo
from test_assessments import report, response
from test_departure import features, context


class AssessmentRefinementTests(unittest.TestCase):
    def run_case(self, empty=False, fail=False, multiple=False):
        original = report(); original['model'] = {'id':'synthetic'}
        dense_times = [0, .6, 1.2, 1.8, 2.4, 2.99]
        dense = parse_candidates(json.dumps(response()), dense_times)
        if empty:
            dense['episodes'] = []
        feature = features()
        feature['feet'] += [[.8,.5]]*3
        feature['handover_active'] += [True]*3
        observer = Mock()
        coarse_response = response()
        if multiple:
            extra = copy.deepcopy(coarse_response['episodes'][0])
            extra.update(inside_frame=1, anchor_frame=2, last_frame=2)
            coarse_response['episodes'].append(extra)
        observer._observe_once.side_effect = [
            parse_candidates(json.dumps(coarse_response), [0,1,2]),
            ModelOutputError('invalid_schema') if fail else dense,
            ModelOutputError('invalid_schema') if fail else feature]
        frames = {i:SampledFrame(i,None) for i in range(3)}
        refined_frames = {target:SampledFrame(actual,None) for target,actual in
                          zip([0,.6,1.2,1.8,2.4,3],dense_times)}
        deadline = time.monotonic()+30
        with patch('app.inference.assessments.probe_video',return_value=VideoInfo(3,0,64,64,0)), \
             patch('app.inference.assessments.sample_frames',side_effect=[frames,refined_frames]):
            bundle = extract_assessments(observer,'synthetic.mov',original,
                                         local_context=context(),deadline=deadline)
        self.assertTrue(all(c.args[2] == deadline for c in observer._observe_once.call_args_list))
        self.assertEqual(observer._observe_once.call_count,3)
        return original,bundle

    def test_empty_coarse_search_does_not_generate_unused_features(self):
        original=report();parsed=parse_candidates(json.dumps(response()),[0,1,2]);parsed['episodes']=[]
        observer=Mock();observer._observe_once.return_value=parsed
        with patch('app.inference.assessments.probe_video',return_value=VideoInfo(3,0,64,64,0)), \
             patch('app.inference.assessments.sample_frames',return_value={i:SampledFrame(i,None) for i in range(3)}):
            bundle=extract_assessments(observer,'synthetic.mov',original)
        self.assertEqual(observer._observe_once.call_count,1)
        self.assertEqual(bundle['assessments'],[])
        self.assertEqual(bundle['diagnostics']['search']['execution_trace'][0]['completed_stages'],
                         ['visual_candidate','predicate_evaluation'])

    def test_multiple_proposals_share_one_identical_refinement_interval(self):
        _,bundle = self.run_case(multiple=True)
        self.assertEqual(bundle['metrics']['refined_windows'],1)
        self.assertEqual(len(bundle['report']['windows']),3)

    def test_distinct_proposals_each_receive_their_own_dense_window(self):
        original=report();original['video']['duration_sec']=10
        original['windows'][0].update(window_end_sec=10,frame_timestamps=[0,4,8])
        raw=response();other=copy.deepcopy(raw['episodes'][0])
        other.update(inside_frame=1,anchor_frame=2,last_frame=2);raw['episodes'].append(other)
        coarse=parse_candidates(json.dumps(raw),[0,4,8])
        sequences=[[2,2.8,3.6,4.4,5.2,6],[6,6.8,7.6,8.4,9.2,9.99]]
        f=features();f['feet'] += [None]*3;f['handover_active'] += [None]*3
        observer=Mock();observer._observe_once.side_effect=[coarse,
            parse_candidates(json.dumps(response()),sequences[0]),f,
            parse_candidates(json.dumps(response()),sequences[1]),f]
        batches=[{t:SampledFrame(t,None) for t in [0,4,8]}]
        for times in sequences:
            targets=times[:-1]+([10] if times[-1]==9.99 else [times[-1]])
            batches.append({t:SampledFrame(actual,None) for t,actual in zip(targets,times)})
        with patch('app.inference.assessments.probe_video',return_value=VideoInfo(10,0,64,64,0)), \
             patch('app.inference.assessments.sample_frames',side_effect=batches):
            bundle=extract_assessments(observer,'synthetic.mov',original)
        self.assertEqual(bundle['metrics']['refined_windows'],2)
        self.assertEqual(len(bundle['assessments']),2)
        self.assertEqual(len({p['event_id'] for p in bundle['assessments']}),2)
        self.assertEqual(bundle['diagnostics']['search']['detector_runs'][0]['processed_window_ids'],[0,1,2,3])

    def test_dense_pts_completion_and_original_windows_preserved(self):
        original,bundle = self.run_case()
        self.assertEqual(bundle['report']['windows'][0], original['windows'][0])
        packet = bundle['assessments'][0]
        self.assertEqual(packet['anchor_sec'],.6)
        self.assertTrue(packet['refinement_complete'])
        self.assertFalse(packet['full_sequence_visible'])
        self.assertTrue(packet['facts']['cashier_left_workplace'])
        self.assertEqual(bundle['report']['windows'][2]['frame_timestamps'][-1],2.99)
        run = bundle['diagnostics']['search']['detector_runs'][0]
        self.assertEqual(run['refinement_status'],'completed')
        self.assertEqual(run['processed_window_ids'],[0,1,2])

    def test_refinement_can_remove_coarse_proposal(self):
        _,bundle = self.run_case(empty=True)
        self.assertEqual(bundle['assessments'],[])
        self.assertEqual(bundle['metrics']['refined_windows'],1)
        self.assertEqual(bundle['diagnostics']['search']['execution_trace'][0]['refinements'][0]['remaining_proposals'],0)

    def test_failed_refinement_is_not_successful_search(self):
        with self.assertRaises(ModelOutputError):
            self.run_case(fail=True)

    @unittest.skipUnless(os.environ.get('FORTE_PEER_SNAPSHOT'),'Set B/C snapshot')
    def test_refined_synthetic_positive_reaches_real_b_c(self):
        import app
        root = Path(os.environ['FORTE_PEER_SNAPSHOT']).resolve()
        app.__path__.append(str(root/'app'))
        rules = importlib.import_module('app.rules')
        detection = importlib.import_module('app.jobs.detection')
        registry = copy.deepcopy(rules.load_registry(root/'config/rules.yaml'))
        for rule in registry['rules']:
            if rule['rule_id']=='cashier_leaves_service':
                rule['status']='reviewed';rule['validated_on_local_cases']=True
                rule['source']['reviewer']='synthetic-only'
        original,bundle = self.run_case()
        decision = detection.finalize_bundle(original,bundle,
                    supported_rule_ids=['cashier_leaves_service'],registry=registry,
                    local_context=context())
        self.assertEqual(decision['verdict'],'violation')
        self.assertEqual(len(decision['supported']),1)


if __name__ == '__main__': unittest.main()
