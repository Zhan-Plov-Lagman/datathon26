"""Synthetic signs-policy checks; no acceptance or accuracy is fabricated."""
from copy import deepcopy

import pytest

from app.inference.targeted import FACTS
from app.jobs.unified import (SignsProcessor, finalize_signs, finalize_unified,
                              create_signs_processor, SIGNS_POLICY_VERSION)
from app.rules.engine import evaluate, evaluate_signs, load_registry, validate_signs_registry
from test_unified import analyzer, movie


def test_signs_processor_uses_source_jpeg_without_fabricating_review(movie):
    registry=load_registry()
    before=deepcopy(registry)
    processor=SignsProcessor(analyzer(),registry)
    output=processor.process(movie,'synthetic-signs',30.)
    assert output['verdict']=='violation'
    assert output['analysis_kind']=='signs_evaluated' and output['rules_reviewed'] is False
    assert output['source_texts_verified'] is True and output['accuracy_status']=='unmeasured'
    assert output['negative_policy_version']==SIGNS_POLICY_VERSION
    event=output['violations'][0]
    assert event['timecode_sec']==event['evidence_frames'][0]['timestamp_sec']==2.9
    assert 'признаки возможного нарушения' in event['description']
    assert registry['rules'][0]['source']['url'] in event['description']
    assert registry==before==processor.registry
    assert all(r['status']=='hypothesis' and r['validated_on_local_cases'] is False for r in registry['rules'])


@pytest.mark.parametrize('mode',['negative','unknown','conflict'])
def test_completed_scoped_unknown_or_conflict_can_return_negative(movie,mode):
    bundle=analyzer(mode).analyze(movie,'synthetic-signs')
    before=deepcopy(bundle)
    result=finalize_signs(bundle,load_registry())
    assert result['verdict']=='no_violation' and result['supported']==[]
    assert result['negative_policy_version']==SIGNS_POLICY_VERSION
    assert bundle==before
    if mode=='conflict':assert result['conflicting_packets']==1
    if mode=='unknown':assert result['candidates'][0]['decision']=='candidate'


def test_incomplete_refinement_cannot_produce_negative(movie):
    bundle=analyzer().analyze(movie,'synthetic-signs')
    bundle['diagnostics']['search']['refinement_runs']=[]
    with pytest.raises(ValueError):finalize_signs(bundle,load_registry())


@pytest.mark.parametrize('mutation',['retired','unverified','document','clause','url','missing_class'])
def test_signs_startup_rejects_missing_verified_source_or_supported_class(mutation):
    registry=load_registry()
    rule=registry['rules'][0]
    if mutation=='retired':rule['status']='retired'
    elif mutation=='unverified':rule['source']['text_verified']=False
    elif mutation=='missing_class':registry['rules'].pop(0)
    else:rule['source'][mutation]=''
    with pytest.raises(ValueError):validate_signs_registry(registry,list(FACTS))


def test_explicit_factory_requires_no_acceptance_file(monkeypatch):
    import app.jobs.unified as module
    monkeypatch.delenv('FORTE_UNIFIED_RELEASE',raising=False)
    monkeypatch.setattr(module,'create_analyzer',analyzer)
    processor=create_signs_processor()
    assert isinstance(processor,SignsProcessor)
    assert not hasattr(processor,'release')


def packet_fixture():
    report={'schema_version':'observations.v1','analysis_kind':'observations_only',
        'complete':True,'coverage_scope':'interval','job_id':'synthetic-signs','video':{'duration_sec':2.},
        'windows':[{'window_id':0,'window_start_sec':0.,'window_end_sec':1.,
            'frame_timestamps':[0.,1.],'visibility_limits':[],
            'observations':[{'start_frame':0,'end_frame':1,'start_sec':0.,'end_sec':1.}]}]}
    packet={'rule_id':'cashier_leaves_service','event_id':'synthetic-event',
        'facts':{name:name!='service_completed' for name in FACTS['cashier_leaves_service']},
        'observation_refs':[{'window_id':0,'observation_index':0}],
        'anchor_sec':1.,'refinement_complete':True,'full_sequence_visible':False}
    return report,packet,load_registry()


def test_default_evaluation_stays_diagnostic_while_explicit_signs_can_qualify():
    report,packet,registry=packet_fixture()
    assert evaluate(report,[packet],registry)['supported']==[]
    assert evaluate(report,[packet],registry)['candidates'][0]['reasons']==['rule_not_reviewed']
    assert len(evaluate_signs(report,[packet],registry,supported_rule_ids=list(FACTS))['supported'])==1


@pytest.mark.parametrize('mutation',['unknown','false_positive','exception','visibility','refinement','sequence'])
def test_signs_preserve_all_existing_evidence_and_predicate_checks(mutation):
    report,packet,registry=packet_fixture()
    if mutation=='unknown':packet['facts']['actor_is_cashier']=None
    elif mutation=='false_positive':packet['facts']['actor_is_cashier']=False
    elif mutation=='exception':packet['facts']['service_completed']=True
    elif mutation=='visibility':report['windows'][0]['visibility_limits']=['Synthetic occlusion']
    elif mutation=='refinement':packet['refinement_complete']=False
    elif mutation=='sequence':registry['rules'][0]['requires_full_sequence']=True
    assert evaluate_signs(report,[packet],registry,supported_rule_ids=list(FACTS))['supported']==[]
