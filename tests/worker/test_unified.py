"""Synthetic fixtures only: real producer, rule adapter and source JPEG path."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import av
import numpy as np
import pytest

from app.api.schema import AnalysisResult
from app.inference.schema import ModelOutputError
from app.inference.targeted import FACTS,parse_facts
from app.inference.unified import UnifiedAnalyzer,fingerprint
from app.jobs.evaluator import registry_digest
from app.jobs.unified import POLICY_VERSION,UnifiedProcessor,finalize_unified,validate_bundle
from app.rules.engine import load_registry
from app.video.sampling import DeadlineExceeded


@pytest.fixture
def movie(tmp_path):
    path=tmp_path/'synthetic.mov'
    with av.open(str(path),'w') as container:
        stream=container.add_stream('mpeg4',rate=10)
        stream.width,stream.height,stream.pix_fmt=160,96,'yuv420p'
        for i in range(30):
            frame=av.VideoFrame.from_ndarray(np.full((96,160,3),i*5,dtype=np.uint8),format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def event(count=6):
    return {'rule_id':'cashier_leaves_service','start_frame':0,'end_frame':count-1,
        'anchor_frame':count-1,'actor_box_1000':None,'description':'Synthetic fixture action',
        'facts':{name:{'value':name!='service_completed','frames':[0,count-1]}
                 for name in FACTS['cashier_leaves_service']}}


class Observer:
    model_id='synthetic/test'
    model_revision='synthetic'
    def __init__(self,mode='positive'):
        self.mode,self.calls=mode,0
        self.deadlines=[]
    def _observe_once(self,frames,context,deadline,**kwargs):
        self.calls+=1
        self.deadlines.append(deadline)
        if self.mode=='failed':
            raise RuntimeError('synthetic inference failure')
        if self.mode=='bad_json' and self.calls==1:
            raise ModelOutputError('invalid_json')
        events=[] if self.mode=='negative' else [event(len(frames))]
        if self.mode=='unknown':
            events[0]['facts']['actor_is_cashier']={'value':None,'frames':[]}
        if self.mode=='conflict':
            events[0]['facts']['service_completed']['value']=True
        if self.mode=='context':
            events[0].update(start_frame=2,end_frame=4,anchor_frame=4)
            for fact in events[0]['facts'].values():fact['frames']=[2,4]
            events[0]['facts']['actor_is_cashier']['frames']=[0,len(frames)-1]
        return kwargs['parser'](json.dumps({'events':events}),[f.timestamp_sec for f in frames])


def analyzer(mode='positive'):
    detector=lambda frames,deadline:[[] for _ in frames]
    return UnifiedAnalyzer(Observer(mode),detector,detector,
        producers={name:{'id':'synthetic/test' if name=='vlm' else 'synthetic/'+name,
                         'revision':'synthetic'} for name in ('vlm','people','objects')})


def reviewed_registry():
    registry=load_registry()
    for rule in registry['rules']:
        if rule['rule_id'] in FACTS:
            rule['status']='reviewed'
            rule['validated_on_local_cases']=True
            rule['source']['reviewer']='synthetic test only'
    return registry


def release(model,registry):
    return {'schema_version':'unified_release.v1','registry_sha256':registry_digest(registry),
        'manifest_sha256':fingerprint(model.manifest),'policy_version':POLICY_VERSION,
        'supported_rule_ids':list(FACTS),'validation_reference':'synthetic unit test only'}


def test_positive_source_jpeg_and_actual_rules(movie):
    model=analyzer()
    registry=reviewed_registry()
    processor=UnifiedProcessor(model,registry,release(model,registry))
    output=AnalysisResult.model_validate(processor.process(movie,'synthetic',30.))
    assert output.verdict=='violation'
    assert len(output.violations)==1
    assert output.violations[0].timecode_sec==2.9
    assert output.violations[0].evidence_frames[0].timestamp_sec==2.9
    assert len(set(model.observer.deadlines))==1
    assert model.observer.calls==2


def test_no_proposals_still_runs_periodic_review(movie):
    model=analyzer('negative')
    bundle=model.analyze(movie,'synthetic')
    validate_bundle(bundle)
    assert model.observer.calls==1
    assert not bundle['assessments']
    assert bundle['metrics']['refinement_calls']==0
    registry=reviewed_registry()
    decision=finalize_unified(bundle,registry,release=release(model,registry))
    assert decision['verdict']=='no_violation'


def test_hypothesis_is_diagnostic_not_public(movie):
    model=analyzer()
    bundle=model.analyze(movie,'synthetic')
    result=finalize_unified(bundle,load_registry())
    assert result['verdict'] is None and result['public_verdict_ready'] is False
    assert result['candidates'][0]['decision']=='candidate'
    with pytest.raises(ValueError):
        UnifiedProcessor(model,load_registry(),release(model,load_registry()))


@pytest.mark.parametrize('mode',['unknown','conflict'])
def test_unknown_or_conflict_does_not_become_supported(movie,mode):
    model=analyzer(mode)
    bundle=model.analyze(movie,'synthetic')
    result=finalize_unified(bundle,reviewed_registry())
    assert not result['supported']
    assert result['candidates'][0]['decision']=='candidate'


@pytest.mark.parametrize('mutation',[
    'missing_review','missing_refinement','wrong_model','wrong_fact_time','missing_scan',
    'forged_stage','duplicate_packet','coarse_fact','wrong_job','fake_coverage'])
def test_incomplete_or_mismatched_execution_is_failure(movie,mutation):
    bundle=analyzer().analyze(movie,'synthetic')
    search=bundle['diagnostics']['search']
    if mutation=='missing_review':
        search['review_runs']=[]
    elif mutation=='missing_refinement':
        search['refinement_runs']=[]
    elif mutation=='wrong_model':
        bundle['report']['manifest']['producers']['vlm']['revision']='other'
    elif mutation=='wrong_fact_time':
        bundle['diagnostics']['episodes'][0]['fact_sources']['actor_is_cashier']['frame_timestamps_sec']=[1.234]
    elif mutation=='missing_scan':
        search['cv']['people_frame_timestamps_sec'].pop()
    elif mutation=='forged_stage':
        search['review_runs'][0]['status']='skipped'
    elif mutation=='duplicate_packet':
        bundle['assessments'].append(deepcopy(bundle['assessments'][0]))
    elif mutation=='coarse_fact':
        bundle['assessments'][0]['observation_refs'][0]['window_id']=0
    elif mutation=='wrong_job':
        search['job_id']='different'
    elif mutation=='fake_coverage':
        bundle['report']['coverage_scope']='full_video'
    with pytest.raises((ValueError,KeyError)):
        validate_bundle(bundle)


def test_schema_retry_and_failure_recovery_share_deadline(movie):
    model=analyzer('bad_json')
    bundle=model.analyze(movie,'synthetic')
    assert bundle['diagnostics']['search']['review_runs'][0]['attempts']==2
    assert len(set(model.observer.deadlines))==1
    model.observer.mode='failed'
    with pytest.raises(RuntimeError):
        model.analyze(movie,'synthetic')
    model.observer.mode='negative'
    assert model.analyze(movie,'synthetic')['complete']
    with pytest.raises(DeadlineExceeded):
        model.analyze(movie,'synthetic',deadline=time.monotonic()-1)
    assert model.analyze(movie,'synthetic')['complete']


@pytest.mark.parametrize('change',['string_bool','out_of_range','unknown_rule','null_with_evidence','unbound_bool'])
def test_strict_model_fact_parser(change):
    e=event()
    if change=='string_bool': e['facts']['actor_is_cashier']['value']='true'
    elif change=='out_of_range': e['anchor_frame']=6
    elif change=='unknown_rule': e['rule_id']='invented'
    elif change=='null_with_evidence': e['facts']['actor_is_cashier']['value']=None
    elif change=='unbound_bool': e['facts']['actor_is_cashier']['frames']=[]
    with pytest.raises(ModelOutputError):
        parse_facts(json.dumps({'events':[e]}),list(range(6)))


@pytest.mark.parametrize('box',[[None],[None,None,None,None],[]])
def test_explicitly_unknown_geometry_normalizes_without_fabricating_coordinates(box):
    e=event()
    e['actor_box_1000']=box
    result=parse_facts(json.dumps({'events':[e]}),list(range(6)))
    assert result['events'][0]['actor_box'] is None
    assert result['null_geometry_normalizations']==1


@pytest.mark.parametrize('box',[[0,0,1001,700],[0,None,1,1],[0,0,1],[-.1,0,1,1]])
def test_known_or_mixed_invalid_geometry_is_not_silently_changed(box):
    e=event()
    e['actor_box_1000']=box
    with pytest.raises(ModelOutputError):
        parse_facts(json.dumps({'events':[e]}),list(range(6)))


def test_explicit_qwen_grid_converts_units_without_clipping():
    e=event()
    e['actor_box_1000']=[100,200,900,800]
    result=parse_facts(json.dumps({'events':[e]}),list(range(6)))
    assert result['events'][0]['actor_box']==[.1,.2,.9,.8]


def test_context_frames_do_not_expand_the_actual_event_interval(movie):
    model=analyzer('context')
    bundle=model.analyze(movie,'synthetic')
    result=finalize_unified(bundle,reviewed_registry())
    event=result['supported'][0]
    episode=bundle['diagnostics']['episodes'][0]
    context=episode['fact_sources']['actor_is_cashier']['frame_timestamps_sec']
    assert context[0]<event['start_sec'] and context[-1]>event['end_sec']
    assert event['start_sec']==episode['start_sec'] and event['end_sec']==episode['end_sec']
    assert event['start_sec']<=event['anchor_sec']<=event['end_sec']


def test_review_coalescing_preserves_full_intervals_actors_and_sampling_density():
    from app.perception.temporal import plan_reviews,coalesce_reviews
    from app.inference.unified import PREPROCESSING
    original=plan_reviews(73.,[(i/3,'actor_motion',i%3) for i in range(219)])
    merged=coalesce_reviews(original,PREPROCESSING['review_span_sec'])
    assert len(merged)<len(original)/5
    assert PREPROCESSING['review_span_sec']/(PREPROCESSING['review_frames']-1)<=2.
    for window in original:
        assert any(m.start_sec<=window.start_sec and m.end_sec>=window.end_sec
            and m.reasons.issuperset(window.reasons) and m.actor_ids.issuperset(window.actor_ids) for m in merged)
