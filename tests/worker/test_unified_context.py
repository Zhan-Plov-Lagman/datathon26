"""Synthetic context/PTS contract tests; no model or private video."""
from copy import deepcopy
import json
import time
from types import SimpleNamespace

import pytest

from app.inference.targeted import FACTS
from app.inference.unified import UnifiedAnalyzer, fingerprint
from app.jobs.unified import validate_bundle, finalize_unified
from app.rules.engine import load_registry


@pytest.fixture
def context_bundle(monkeypatch):
    import app.inference.unified as module
    duration=20.
    scan={'schema_version':'perception.v1','complete':True,'coverage_scope':'full_video_sampled',
          'job_id':'synthetic-context','duration_sec':duration,'sample_fps':3.,'object_period_sec':5.,
          'samples':[{'pts':i/3,'actors':[]} for i in range(61)],
          'object_samples':[{'pts':float(i)} for i in range(0,21,5)],
          'review_windows':[{'start_sec':0.,'end_sec':20.}],
          'metrics':{'elapsed_sec':0.,'people_frames':61,'object_frames':5,'original_review_windows':1}}
    monkeypatch.setattr(module,'scan_video',lambda *a,**k:deepcopy(scan))
    monkeypatch.setattr(module,'probe_video',lambda *a:SimpleNamespace(duration_sec=duration))
    monkeypatch.setattr(module,'sample_frames',lambda path,targets,info,**k:
        {t:SimpleNamespace(timestamp_sec=t,image=None) for t in targets})
    class Observer:
        model_id='synthetic/test'
        model_revision='synthetic'
        def __init__(self):self.calls=0;self.deadlines=[]
        def _observe_once(self,frames,context,deadline,**kwargs):
            self.calls+=1;self.deadlines.append(deadline)
            pts=[f.timestamp_sec for f in frames]
            anchor=min(range(len(pts)),key=lambda i:abs(pts[i]-10.9))
            e={'rule_id':'cashier_leaves_service','start_frame':anchor,'end_frame':anchor,
               'anchor_frame':anchor,'actor_box_1000':None,'description':'Synthetic fixture',
               'facts':{name:{'value':None,'frames':[]} for name in FACTS['cashier_leaves_service']}}
            e['facts']['actor_is_cashier']={'value':True,'frames':[0]}
            return kwargs['parser'](json.dumps({'events':[e]}),pts)
    model=UnifiedAnalyzer(Observer(),None,None,producers={name:{'id':'synthetic/test',
        'revision':'synthetic'} for name in ('vlm','people','objects')})
    bundle=model.analyze('synthetic-only','synthetic-context')
    assert model.observer.calls==2 and len(set(model.observer.deadlines))==1
    return bundle


def test_refinement_keeps_dense_targets_and_parent_context(context_bundle):
    bundle=context_bundle
    validate_bundle(bundle)
    run=bundle['diagnostics']['search']['refinement_runs'][0]
    assert len(run['scheduled_targets_sec'])==len(run['dense_frame_timestamps_sec'])==6
    assert 0<len(run['context_frame_timestamps_sec'])<=6
    assert len(run['frame_timestamps_sec'])<=12
    assert run['context_frame_timestamps_sec'][0]<run['dense_frame_timestamps_sec'][0]
    assert run['context_frame_timestamps_sec'][-1]>run['dense_frame_timestamps_sec'][-1]
    assert not bundle['assessments'][0]['full_sequence_visible']
    episode=bundle['diagnostics']['episodes'][0]
    assert episode['fact_sources']['actor_is_cashier']['frame_timestamps_sec'][0]<episode['start_sec']
    decision=finalize_unified(bundle,load_registry())
    assert decision['candidates'][0]['start_sec']==episode['start_sec']
    assert decision['candidates'][0]['end_sec']==episode['end_sec']


@pytest.mark.parametrize('field',['scheduled_targets_sec','scheduled_frame_timestamps_sec',
    'dense_frame_timestamps_sec','context_frame_timestamps_sec','parent_window_ids'])
def test_context_provenance_tampering_rejected(context_bundle,field):
    bundle=deepcopy(context_bundle)
    bundle['diagnostics']['search']['refinement_runs'][0][field]=[]
    with pytest.raises(ValueError):validate_bundle(bundle)


def test_legacy_v13_without_context_still_validates(context_bundle):
    # No proposals permits a legacy receipt without newly introduced fields.
    bundle=deepcopy(context_bundle)
    bundle['report']['manifest']['pipeline_version']='cv-targeted.v1.3'
    bundle['report']['manifest']['preprocessing'].pop('refine_context_frames')
    bundle['report']['windows']=bundle['report']['windows'][:1]
    bundle['report']['windows'][0]['observations']=[]
    bundle['assessments']=[]
    bundle['diagnostics']['episodes']=[]
    search=bundle['diagnostics']['search']
    search['refinement_runs']=[]
    search['planned_refinement_intervals']=[]
    search['manifest_sha256']=fingerprint(bundle['report']['manifest'])
    validate_bundle(bundle)


@pytest.mark.parametrize('dense_indices,expected_calls',[(None,1),([1],2)])
def test_review_selects_vllm_fact_grammar_and_retries_invalid_dense_anchor(dense_indices,expected_calls):
    from app.inference.targeted import review
    from app.inference.vllm_observer import VllmObserver
    observer=object.__new__(VllmObserver)
    observer.max_new_tokens=128
    observer.processor=SimpleNamespace(apply_chat_template=lambda *a,**k:'synthetic prompt')
    observer.SamplingParams=lambda **kwargs:kwargs
    observer.StructuredOutputsParams=lambda **kwargs:kwargs
    calls=[]
    def generate(inputs,**kwargs):
        anchor=len(calls)
        calls.append(kwargs)
        event={'rule_id':'cashier_leaves_service','start_frame':anchor,'end_frame':anchor,
               'anchor_frame':anchor,'actor_box_1000':None,'description':'Synthetic fixture',
               'facts':{name:{'value':None,'frames':[]} for name in FACTS['cashier_leaves_service']}}
        return [SimpleNamespace(outputs=[SimpleNamespace(text=json.dumps({'events':[event]}),
            token_ids=[1],finish_reason='stop')])]
    observer.model=SimpleNamespace(generate=generate)
    frames=[SimpleNamespace(image=object(),timestamp_sec=t) for t in (0.,1.)]
    result=review(observer,frames,time.monotonic()+10,dense_indices=dense_indices)
    assert len(calls)==result['attempts']==expected_calls
    assert result['events'][0]['anchor_frame']==expected_calls-1
    assert all('json' in call['sampling_params']['structured_outputs'] for call in calls)
