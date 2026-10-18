"""Synthetic contract checks; these do not claim real vLLM/model accuracy."""
import time
from types import SimpleNamespace

import pytest

from app.inference.schema import ModelOutputError
from app.inference.targeted import parse_facts
from app.inference.vllm_observer import VllmObserver, facts_schema, local_model_info
from app.video.sampling import DeadlineExceeded


def fake_observer(text='{"events":[]}'):
    observer=object.__new__(VllmObserver)
    observer.max_new_tokens=128
    observer.calls=[]
    observer.processor=SimpleNamespace(apply_chat_template=lambda messages,**kwargs:repr(messages))
    observer.SamplingParams=lambda **kwargs:kwargs
    observer.StructuredOutputsParams=lambda **kwargs:kwargs
    def generate(inputs,**kwargs):
        observer.calls.append((inputs,kwargs))
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text,token_ids=[1,2],finish_reason='stop')])]
    observer.model=SimpleNamespace(generate=generate)
    return observer


def test_vllm_reuses_actual_images_and_pts_with_json_grammar():
    observer=fake_observer()
    frames=[SimpleNamespace(image=object(),timestamp_sec=t) for t in (1.2,1.7)]
    result=observer._observe_once(frames,'',time.monotonic()+10,system_prompt='Synthetic.',parser=parse_facts)
    assert result['events']==[]
    inputs,options=observer.calls[0]
    assert inputs[0]['multi_modal_data']['image']==[f.image for f in frames]
    assert 'Image index 1:' in inputs[0]['prompt']
    schema=options['sampling_params']['structured_outputs']['json']
    assert schema['required']==['events']
    assert schema['properties']['events']['maxItems']==2
    assert schema['properties']['events']['items']['anyOf'][0]['properties']['start_frame']['maximum']==1
    assert options['use_tqdm'] is False


def test_vllm_invalid_schema_remains_failure_and_private_sink():
    observer=fake_observer('{"bad":true}')
    saved=[]
    observer._invalid_response_sink=lambda raw,metadata:saved.append((raw,metadata))
    with pytest.raises(ModelOutputError) as error:
        observer._observe_once([SimpleNamespace(image=object(),timestamp_sec=0.)],'',time.monotonic()+10,parser=parse_facts)
    assert error.value.generated_tokens==2
    assert saved[0][1]['reason']=='invalid_schema'


def test_vllm_expired_deadline_never_runs():
    observer=fake_observer()
    with pytest.raises(DeadlineExceeded):
        observer._observe_once([SimpleNamespace(image=object(),timestamp_sec=0.)],'',time.monotonic()-1,parser=parse_facts)
    assert observer.calls==[]


def test_vllm_late_generation_is_failure():
    observer=fake_observer()
    original=observer.model.generate
    def late(*args,**kwargs):
        time.sleep(.03)
        return original(*args,**kwargs)
    observer.model.generate=late
    with pytest.raises(DeadlineExceeded):
        observer._observe_once([SimpleNamespace(image=object(),timestamp_sec=0.)],'',time.monotonic()+.01,parser=parse_facts)


@pytest.mark.parametrize('count',[0,19])
def test_vllm_frame_cap_prevents_generation(count):
    observer=fake_observer()
    with pytest.raises(ValueError):
        observer._observe_once([SimpleNamespace(image=object(),timestamp_sec=0.)]*count,'',time.monotonic()+10,parser=parse_facts)
    assert observer.calls==[]


@pytest.mark.parametrize('reason',['length','abort'])
def test_vllm_valid_looking_incomplete_output_fails(reason):
    observer=fake_observer()
    observer.model.generate=lambda *a,**kw:[SimpleNamespace(outputs=[
        SimpleNamespace(text='{"events":[]}',token_ids=[1],finish_reason=reason)])]
    with pytest.raises(ModelOutputError):
        observer._observe_once([SimpleNamespace(image=object(),timestamp_sec=0.)],'',time.monotonic()+10,parser=parse_facts)


def test_local_model_requires_pinned_fp8_manifest(tmp_path):
    import json
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'qwen3_vl',
        'quantization_config':{'quant_method':'fp8'}}))
    manifest=tmp_path/'download_manifest.json'
    manifest.write_text(json.dumps({'repo_id':'Qwen/synthetic','revision':'main'}))
    with pytest.raises(ValueError):
        local_model_info(tmp_path)
    manifest.write_text(json.dumps({'repo_id':'Qwen/synthetic','revision':'a'*40}))
    assert local_model_info(tmp_path)[1]['revision']=='a'*40


def test_schema_allows_unknown_fact_and_rejects_extra_key():
    import json
    schema=facts_schema(2)
    branch=schema['properties']['events']['items']['anyOf'][0]['properties']
    event={'rule_id':next(iter(branch['rule_id']['enum'])),
           'start_frame':0,'end_frame':1,'anchor_frame':1,'actor_box_1000':None,
           'description':'synthetic',
           'facts':{name:{'value':None,'frames':[]} for name in branch['facts']['required']}}
    assert schema['additionalProperties'] is False
    assert parse_facts(json.dumps({'events':[event]}),[0.,1.])['events'][0]['facts']==event['facts']
    event['extra']=True
    with pytest.raises(ModelOutputError):
        parse_facts(json.dumps({'events':[event]}),[0.,1.])
