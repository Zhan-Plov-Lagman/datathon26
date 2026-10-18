"""Synthetic release only; the on-disk hypothesis registry is never modified."""
import copy
import time
from unittest.mock import Mock

import pytest

from app.api.schema import AnalysisResult
from app.jobs.detection import finalize_bundle
from app.jobs.evaluator import DetectionEvaluator, registry_digest
from app.jobs.pipeline import IntegratedProcessor
from app.rules import load_registry
from tests.rules.test_detection_bridge import setup_bundle

from app.inference.assessments import extract_assessments
from app.inference.departure import VERSION
from app.video.sampling import SampledFrame, VideoInfo
from unittest.mock import patch


def release_for(registry):
    return dict(validation_reference='synthetic-test-only', registry_sha256=registry_digest(registry),
                model_id='synthetic', model_revision='test',
                detector_versions={'cashier_leaves_service': 'synthetic'}, policy_version='detection-policy.v1')


def evaluate_bundle(bundle, *, context=None):
    original, _, registry, scope = setup_bundle()
    return finalize_bundle(original, bundle, registry=registry,
                           supported_rule_ids=scope, local_context=context)


@pytest.mark.parametrize('damage', ['missing_sources', 'unknown_bool', 'bad_ref', 'outside_packet',
                                    'missing_episode', 'duplicate_episode', 'context_absent', 'blank_reason'])
def test_bad_provenance_does_not_become_negative(damage):
    _, bundle, _, _ = setup_bundle()
    episode = bundle['diagnostics']['episodes'][0]
    source = episode['fact_sources']['service_completed']
    if damage == 'missing_sources':
        del episode['fact_sources']['service_completed']
    elif damage == 'unknown_bool':
        source['origin'] = 'unknown'
    elif damage == 'bad_ref':
        source['observation_refs'][0]['observation_index'] = 99
    elif damage == 'outside_packet':
        window = copy.deepcopy(bundle['report']['windows'][0])
        window['window_id'] = 1
        bundle['report']['windows'].append(window)
        source['observation_refs'][0]['window_id'] = 1
    elif damage == 'missing_episode':
        bundle['diagnostics']['episodes'] = []
    elif damage == 'duplicate_episode':
        bundle['diagnostics']['episodes'].append(copy.deepcopy(episode))
    elif damage == 'context_absent':
        source.update(origin='mixed', context_ref={'view_id': 'invented'})
    else:
        bundle['assessments'][0]['facts']['service_completed'] = None
        source.update(origin='unknown', reason=' ')
    with pytest.raises(ValueError, match='provenance'):
        evaluate_bundle(bundle)


def test_conflict_is_preserved_without_mutating_facts():
    _, bundle, _, _ = setup_bundle()
    bundle['diagnostics']['episodes'][0]['conflicts'] = ['synthetic_source_disagreement']
    before = copy.deepcopy(bundle)
    result = evaluate_bundle(bundle)
    assert result['verdict'] == 'no_violation'
    assert 'synthetic_source_disagreement' in result['unresolved'][0]['reasons']
    assert before == bundle


def test_mixed_context_must_match_supplied_review_and_job():
    original, bundle, registry, scope = setup_bundle()
    original['video'].update(width=64, height=64, rotation=0)
    bundle['report']['video'] = copy.deepcopy(original['video'])
    context = dict(schema_version='local_context.draft.v1', view_id='synthetic-view', context_version='test',
                   reviewer='test', reviewed_at='test', basis_ref='test',
                   coordinate_space='normalized_display_after_rotation',
                   scope=dict(job_id=original['job_id'], display_width=64, display_height=64, rotation=0))
    source = bundle['diagnostics']['episodes'][0]['fact_sources']['actor_is_cashier']
    source.update(origin='mixed', context_ref={key: context[key] for key in
                  ('view_id', 'context_version', 'basis_ref', 'reviewer')})
    def run():
        return finalize_bundle(original, bundle, registry=registry, supported_rule_ids=scope, local_context=context)
    assert run()['verdict'] == 'violation'
    source['context_ref']['context_version'] = 'other'
    with pytest.raises(ValueError):
        run()
    source['context_ref']['context_version'] = 'test'
    context['scope']['job_id'] = 'other'
    with pytest.raises(ValueError):
        run()


@pytest.mark.parametrize('unknown', [False, True])
def test_real_b_policy_to_ruledecision_to_jpeg_and_public_schema(result, unknown):
    report, bundle, registry, _ = setup_bundle()
    if unknown:
        bundle['assessments'][0]['facts']['service_completed'] = None
        bundle['diagnostics']['episodes'][0]['fact_sources']['service_completed'] = {
            'origin': 'unknown', 'observation_refs': [], 'reason': 'synthetic unknown'}
    worker = Mock()
    worker.analyze.return_value = report
    worker.assess.return_value = bundle
    frame = copy.deepcopy(result['violations'][0]['evidence_frames'][0])
    frame['timestamp_sec'] = 2.0
    worker.evidence.return_value = {'0': [frame]}
    evaluator = DetectionEvaluator(registry=registry, release=release_for(registry))
    internal = IntegratedProcessor(worker, evaluator).process('synthetic.mov', 'synthetic-job', 30)
    public = AnalysisResult.model_validate(internal).public('synthetic-job')
    assert set(public) == {'job_id', 'status', 'verdict', 'violations'}
    assert public['verdict'] == ('no_violation' if unknown else 'violation')
    assert worker.assess.call_args.kwargs['deadline'] == worker.analyze.call_args.kwargs['deadline']
    assert 0 < worker.assess.call_args.kwargs['timeout_sec'] < 30
    if unknown:
        worker.evidence.assert_not_called()
    else:
        assert 'Synthetic action' in public['violations'][0]['description']
        assert registry['rules'][0]['source']['url'] in public['violations'][0]['description']
        worker.evidence.assert_called_once()
        assert worker.evidence.call_args.args[1] == {'0': [2.0]}
        # An unrelated JPEG with a valid timestamp inside the video is not sufficient.
        worker.evidence.return_value['0'][0]['timestamp_sec'] = 3.0
        with pytest.raises(ValueError, match='requested source PTS'):
            IntegratedProcessor(worker, evaluator).process('synthetic.mov', 'synthetic-job', 30)


def test_real_registry_still_blocks_startup():
    registry = load_registry()
    with pytest.raises(ValueError):
        DetectionEvaluator(registry=registry, release=release_for(registry))


@pytest.mark.parametrize('mismatch', ['registry', 'model', 'detector', 'policy', 'deadline'])
def test_release_and_deadline_guards(mismatch):
    report, bundle, registry, _ = setup_bundle()
    release = release_for(registry)
    worker = Mock()
    worker.assess.return_value = bundle
    if mismatch == 'registry':
        release['registry_sha256'] = '0' * 64
    if mismatch == 'policy':
        release['policy_version'] = 'other'
    if mismatch in ('registry', 'policy'):
        with pytest.raises(ValueError):
            DetectionEvaluator(registry=registry, release=release)
        return
    evaluator = DetectionEvaluator(registry=registry, release=release)
    if mismatch == 'model':
        report['model']['revision'] = 'other'
    if mismatch == 'detector':
        bundle['diagnostics']['search']['detector_runs'][0]['detector_version'] = 'other'
    deadline = time.monotonic() + (1 if mismatch == 'deadline' else 30)
    with pytest.raises((ValueError, TimeoutError)):
        evaluator.evaluate(report, video_path='synthetic.mov', worker=worker, deadline=deadline)
    if mismatch in ('model', 'deadline'):
        worker.assess.assert_not_called()


@pytest.mark.parametrize('with_context', [True, False])
def test_actual_refinement_through_evaluator_and_source_evidence(result, with_context):
    """Run real A refinement and B/C adapters; only model/decoder are synthetic."""
    report, _, registry, _ = setup_bundle()
    report['video'] = dict(duration_sec=3, width=64, height=64, rotation=0, start_time_sec=0)
    report['windows'] = [dict(window_id=0, window_start_sec=0, window_end_sec=3,
                             frame_timestamps=[0, 1, 2], visibility_limits=[], observations=[])]
    context = dict(schema_version='local_context.draft.v1', view_id='synthetic-view', context_version='test',
                   reviewer='synthetic', reviewed_at='test', basis_ref='synthetic-only',
                   coordinate_space='normalized_display_after_rotation',
                   scope=dict(job_id=report['job_id'], display_width=64, display_height=64, rotation=0),
                   workplace_polygon=[[.1,.1],[.5,.1],[.5,.9],[.1,.9]],
                   adjacent_equipment_included=False, max_transition_gap_sec=1.1,
                   cash_context_confirmation=dict(value=True, basis_ref='synthetic-cash'),
                   functional_role_confirmation=dict(value=True, basis_ref='synthetic-role'))
    raw_candidate = ('{"operation_context":"Synthetic","visibility_limits":[],"observations":'
                     '[{"start_frame":0,"end_frame":2,"description":"Synthetic outward motion"}],'
                     '"episodes":[{"inside_frame":0,"anchor_frame":1,"last_frame":2,"observation_indices":[0]}]}')
    raw_features = ('{"same_person":true,"cash_from_customer":true,"cash_handling":true,'
                    '"visible_crossing":true,"feet":[[0.2,0.5],[0.8,0.5],[0.8,0.5],[0.8,0.5],[0.8,0.5],[0.8,0.5]],'
                    '"handover_active":[true,true,true,true,true,true]}')
    calls = []
    def observe(frames, previous, deadline, **kwargs):
        calls.append(deadline)
        raw = raw_features if len(calls) == 3 else raw_candidate
        return kwargs['parser'](raw, [f.timestamp_sec for f in frames])
    observer = Mock()
    observer._observe_once.side_effect = observe
    dense_times = [0, .6, 1.2, 1.8, 2.4, 2.99]
    batches = [{t: SampledFrame(t, None) for t in [0,1,2]},
               {target: SampledFrame(actual, None) for target, actual in
                zip([0,.6,1.2,1.8,2.4,3], dense_times)}]
    worker = Mock()
    worker.analyze.return_value = report
    worker.assess.side_effect = lambda path, observed, **kw: extract_assessments(observer, path, observed, **kw)
    frame = copy.deepcopy(result['violations'][0]['evidence_frames'][0])
    frame['timestamp_sec'] = .6
    worker.evidence.return_value = {'0': [frame]}
    release = release_for(registry)
    release['detector_versions']['cashier_leaves_service'] = VERSION
    evaluator = DetectionEvaluator(registry=registry, release=release,
                                   context_provider=(lambda _: context) if with_context else None)
    with patch('app.inference.assessments.probe_video', return_value=VideoInfo(3,0,64,64,0)), \
         patch('app.inference.assessments.sample_frames', side_effect=batches):
        completed = IntegratedProcessor(worker, evaluator).process('synthetic.mov', report['job_id'], 30)
    assert completed['verdict'] == ('violation' if with_context else 'no_violation')
    outer_deadline = worker.analyze.call_args.kwargs['deadline']
    assert worker.assess.call_args.kwargs['deadline'] == outer_deadline
    # A takes min(outer deadline, start + timeout): C's JPEG reserve shortens
    # the inner budget, and every refinement stage must respect that budget.
    assert len(calls) == 3 and len(set(calls)) == 1
    assert calls[0] < outer_deadline
    if with_context:
        assert completed['violations'][0]['timecode_sec'] == .6
        assert 'Synthetic outward motion' in completed['violations'][0]['description']
        assert worker.evidence.call_args.args[1] == {'0': [.6]}
    else:
        worker.evidence.assert_not_called()
