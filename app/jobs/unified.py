"""Versioned CV/VLM adapter to the existing rule engine and jury API.

The v1 coordinator remains the baseline. A v2 receipt never pretends that the
VLM inspected every frame of the CV scan. Activation requires a measured release.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
from threading import Lock
import time

from app.api.schema import AnalysisResult
from app.evidence.frames import extract_evidence_batch
from app.inference.targeted import FACTS
from app.inference.unified import PIPELINE_VERSION, UnifiedAnalyzer, fingerprint, _targets, _context_pts
from app.inference.worker import WorkerBusy
from app.jobs.evaluator import registry_digest
from app.rules.engine import evaluate, load_registry, validate_registry, validate_report
from app.video.sampling import check_deadline


POLICY_VERSION = 'detection-targeted.v2'


def require(condition):
    if not condition:
        raise ValueError('Invalid unified execution contract; inspect locally')


def _timeline(values, duration):
    require(isinstance(values,list) and bool(values))
    require(all(type(x) in (float,int) and math.isfinite(x) and 0<=x<=duration for x in values))
    require(all(a<b for a,b in zip(values,values[1:])))


def validate_bundle(bundle):
    require(isinstance(bundle,dict) and bundle.get('schema_version')=='assessment_bundle.v2'
            and bundle.get('complete') is True)
    report = bundle['report']
    require(report.get('schema_version')=='observations.v2' and report.get('complete') is True
            and report.get('coverage_scope')=='cv_full_video_vlm_targeted')
    require(isinstance(report.get('job_id'),str) and bool(report['job_id']))
    duration = report['video']['duration_sec']
    require(type(duration) in (int,float) and math.isfinite(duration) and duration>0)
    # Explicit projection: v1 "interval" describes only the selected VLM frames.
    # Full-video sampled CV coverage is validated separately below.
    projected = deepcopy(report)
    projected['schema_version']='observations.v1'
    projected['coverage_scope']='interval'
    for window in projected['windows']:
        window['window_start_sec']=window['start_sec']
        window['window_end_sec']=window['end_sec']
        window['frame_timestamps']=list(window['frame_timestamps_sec'])
    by_id = validate_report(projected)
    scan = bundle['perception']
    require(scan.get('schema_version')=='perception.v1' and scan.get('complete') is True
        and scan.get('coverage_scope')=='full_video_sampled'
        and scan.get('job_id')==report['job_id'] and scan.get('duration_sec')==duration)
    search = bundle['diagnostics']['search']
    require(search.get('schema_version')=='detector_execution.v2'
        and search.get('job_id')==report['job_id']
        and search.get('manifest_sha256')==fingerprint(report['manifest'])
        and search.get('supported_rule_ids')==list(FACTS))
    cfg = report['manifest']['preprocessing']
    with_context = report['manifest'].get('pipeline_version')=='cv-targeted.v1.4'
    if with_context:
        require(cfg.get('refine_context_frames')==6)
    producers=report['manifest']['producers']
    require(set(producers)=={'vlm','people','objects'})
    require(all(report['model'][k]==producers['vlm'][k] for k in ('id','revision')))
    require(scan['sample_fps']==cfg['cv_fps'] and scan['object_period_sec']==cfg['object_period_sec'])
    cv = search['cv']
    require(cv.get('status')=='complete' and cv.get('coverage_scope')=='full_video_sampled')
    for key, source, period in (
        ('people_frame_timestamps_sec','samples',1/scan['sample_fps']),
        ('object_frame_timestamps_sec','object_samples',scan['object_period_sec'])):
        pts = cv[key]
        _timeline(pts,duration)
        require(pts==[s['pts'] for s in scan[source]])
        require(pts[0]<=.5 and duration-pts[-1]<=period+.5)
        require(all(b-a<=period+.5 for a,b in zip(pts,pts[1:])))
    require(search['planned_review_windows']==scan['review_windows'])
    review_runs,refine_runs = search['review_runs'],search['refinement_runs']
    require(isinstance(review_runs,list) and len(review_runs)>0
            and len(review_runs)==len(scan['review_windows']))
    require(isinstance(refine_runs,list) and len(refine_runs)==len(search['planned_refinement_intervals']))
    seen=set()
    for stage, runs, intervals in (
        ('proposal',review_runs,[(w['start_sec'],w['end_sec']) for w in scan['review_windows']]),
        ('refinement',refine_runs,search['planned_refinement_intervals'])):
        for run, interval in zip(runs,intervals,strict=True):
            wid=run.get('window_id')
            require(type(wid) is int and wid in by_id and wid not in seen)
            seen.add(wid)
            require(run.get('status')=='complete' and type(run.get('attempts')) is int and 1<=run['attempts']<=2)
            window=by_id[wid]
            require(window.get('stage')==stage and run['frame_timestamps_sec']==window['frame_timestamps_sec'])
            pts=run['frame_timestamps_sec']
            _timeline(pts,duration)
            require(len(interval)==2 and 0<=interval[0]<interval[1]<=duration)
            if stage=='refinement' and with_context:
                targets=_targets(*interval,cfg['refine_frames'])
                require(run.get('scheduled_targets_sec')==targets)
                scheduled=run.get('scheduled_frame_timestamps_sec')
                require(isinstance(scheduled,list) and len(scheduled)==len(targets))
                require(all(type(t) in (int,float) and math.isfinite(t) and 0<=t<=duration
                            and abs(t-target)<=.5 for t,target in zip(scheduled,targets,strict=True)))
                require(all(a<=b for a,b in zip(scheduled,scheduled[1:])))
                dense=run.get('dense_frame_timestamps_sec')
                require(dense==sorted(set(scheduled)))
                context=run.get('context_frame_timestamps_sec')
                parents=run.get('parent_window_ids')
                require(isinstance(parents,list) and bool(parents) and len(parents)==len(set(parents)))
                require(all(type(p) is int and p in by_id and by_id[p]['stage']=='proposal' for p in parents))
                expected_context=_context_pts([by_id[p]['frame_timestamps_sec'] for p in parents],
                                              interval,cfg['refine_context_frames'])
                require(context==[t for t in expected_context if t not in dense])
                require(pts==sorted(set(dense+context)))
                pts=dense
            require(abs(pts[0]-interval[0])<=.5 and abs(pts[-1]-interval[1])<=.5)
            count=cfg['review_frames'] if stage=='proposal' else cfg['refine_frames']
            require(len(pts)<=count)
            max_gap=(interval[1]-interval[0])/(count-1)+.5
            require(all(b-a<=max_gap for a,b in zip(pts,pts[1:])))
    require(seen==set(by_id))
    expected_refinements=[]
    expected_parents={}
    for run in review_runs:
        for observation in by_id[run['window_id']]['observations']:
            anchor=observation['anchor_sec']
            require(anchor in by_id[run['window_id']]['frame_timestamps_sec'])
            start=max(0.,anchor-cfg['refine_sec']/2)
            end=min(duration,start+cfg['refine_sec'])
            pair=[round(max(0.,end-cfg['refine_sec']),6),round(end,6)]
            if pair not in expected_refinements:
                expected_refinements.append(pair)
                expected_parents[tuple(pair)]=[]
            if run['window_id'] not in expected_parents[tuple(pair)]:
                expected_parents[tuple(pair)].append(run['window_id'])
    require(search['planned_refinement_intervals']==expected_refinements)
    if with_context:
        for run,interval in zip(refine_runs,expected_refinements,strict=True):
            require(run['parent_window_ids']==expected_parents[tuple(interval)])
    packets = bundle['assessments']
    episodes = bundle['diagnostics']['episodes']
    require(isinstance(packets,list) and isinstance(episodes,list) and len(packets)==len(episodes))
    episode_map={}
    for episode in episodes:
        key=(episode['rule_id'],episode['event_id'])
        require(key not in episode_map)
        episode_map[key]=episode
    packet_keys=set()
    referenced_observations=set()
    for packet in packets:
        key=(packet['rule_id'],packet['event_id'])
        require(key in episode_map and key not in packet_keys and key[0] in FACTS)
        packet_keys.add(key)
        require(packet.get('refinement_complete') is True and packet.get('full_sequence_visible') is False)
        require(set(packet['facts'])==set(FACTS[key[0]]))
        refs=packet['observation_refs']
        require(isinstance(refs,list) and bool(refs))
        wid,idx=refs[0]['window_id'],refs[0]['observation_index']
        require(type(wid) is int and type(idx) is int and wid in by_id
                and by_id[wid]['stage']=='refinement' and 0<=idx<len(by_id[wid]['observations']))
        observation=by_id[wid]['observations'][idx]
        require('feature_key' not in observation)
        for ref in refs:
            require(ref['window_id']==wid and type(ref['observation_index']) is int
                    and 0<=ref['observation_index']<len(by_id[wid]['observations']))
            identity=(wid,ref['observation_index'])
            require(identity not in referenced_observations)
            referenced_observations.add(identity)
        require(observation['rule_id']==packet['rule_id'] and observation['anchor_sec']==packet['anchor_sec'])
        require(packet['anchor_sec'] in by_id[wid]['frame_timestamps_sec']
            and observation['start_sec']<=packet['anchor_sec']<=observation['end_sec'])
        if with_context:
            run=next(r for r in refine_runs if r['window_id']==wid)
            require(packet['anchor_sec'] in run['dense_frame_timestamps_sec'])
        episode=episode_map[key]
        require(episode['start_sec']==observation['start_sec'] and episode['end_sec']==observation['end_sec'])
        require(set(episode['fact_sources'])==set(packet['facts']))
        for name,value in packet['facts'].items():
            source=episode['fact_sources'][name]
            require(value is None or type(value) is bool)
            require(isinstance(source.get('reason'),str) and bool(source['reason']))
            if value is None:
                require(source['origin']=='unknown' and not source['observation_refs']
                        and not source['frame_timestamps_sec'])
            else:
                feature_refs=source['observation_refs']
                require(source['origin']=='visual' and isinstance(feature_refs,list) and len(feature_refs)==1
                        and feature_refs[0] in refs[1:])
                feature=by_id[wid]['observations'][feature_refs[0]['observation_index']]
                require(feature.get('feature_key')==name and feature.get('value') is value)
                pts=source['frame_timestamps_sec']
                _timeline(pts,duration)
                require(all(t in by_id[wid]['frame_timestamps_sec'] and
                    feature['start_sec']<=t<=feature['end_sec'] for t in pts))
                require(feature['start_sec']==pts[0] and feature['end_sec']==pts[-1])
    require(referenced_observations=={(r['window_id'],i) for r in refine_runs
        for i in range(len(by_id[r['window_id']]['observations']))})
    return projected,episode_map


def finalize_unified(bundle, registry, *, release=None):
    """No release: diagnostic candidates only. Validated release: scoped two-verdict policy."""
    report,episodes = validate_bundle(bundle)
    validate_registry(registry)
    if release is not None:
        validate_release(release,registry,bundle['report']['manifest'])
    packets=deepcopy(bundle['assessments'])
    conflicts=[]
    for packet in packets:
        facts=packet['facts']
        if facts.get('service_in_progress') is True and facts.get('service_completed') is True:
            conflicts.append((packet['rule_id'],packet['event_id']))
            packet['facts']=dict.fromkeys(facts)
    result=evaluate(report,packets,registry)
    # Supporting context may precede the event. Preserve the event's own bounds.
    for event in result['candidates']:
        episode=episodes[(event['rule_id'],event['event_id'])]
        event['start_sec'],event['end_sec']=episode['start_sec'],episode['end_sec']
    supported=[]
    duplicate_count=0
    for event in result['supported']:
        ep=episodes[(event['rule_id'],event['event_id'])]
        duplicate=False
        for previous in supported:
            old=episodes[(previous['rule_id'],previous['event_id'])]
            if (previous['rule_id']==event['rule_id'] and ep['actor_track_id'] is not None
                and ep['actor_track_id']==old['actor_track_id']
                and max(ep['start_sec'],old['start_sec'])<=min(ep['end_sec'],old['end_sec'])):
                duplicate=True
                break
        if duplicate:
            duplicate_count+=1
        else:
            supported.append(event)
    result.update({'schema_version':'detection_result.v2','policy_version':POLICY_VERSION,
        'public_verdict_ready':release is not None, 'supported':supported,
        'verdict':('violation' if supported else 'no_violation') if release is not None else None,
        'negative_policy_version':POLICY_VERSION if release is not None else None,
        'duplicate_events_removed':duplicate_count,'conflicting_packets':len(conflicts),
        'manifest_sha256':fingerprint(bundle['report']['manifest'])})
    return result


def validate_release(release, registry, manifest):
    require(isinstance(release,dict) and release.get('schema_version')=='unified_release.v1')
    require(release.get('registry_sha256')==registry_digest(registry)
        and release.get('manifest_sha256')==fingerprint(manifest)
        and release.get('policy_version')==POLICY_VERSION
        and release.get('supported_rule_ids')==list(FACTS))
    require(isinstance(release.get('validation_reference'),str) and bool(release['validation_reference'].strip()))
    rules={r['rule_id']:r for r in registry['rules']}
    require(all(r in rules and rules[r]['status']=='reviewed'
        and rules[r].get('validated_on_local_cases') is True for r in FACTS))


class UnifiedProcessor:
    pipeline_version=PIPELINE_VERSION

    def __init__(self, analyzer, registry, release):
        validate_registry(registry)
        validate_release(release,registry,analyzer.manifest)
        self.analyzer,self.registry,self.release=analyzer,deepcopy(registry),deepcopy(release)
        self.pipeline_version=PIPELINE_VERSION+'+'+fingerprint(analyzer.manifest)[:16]
        self._lock=Lock()

    def process(self,video_path,job_id,timeout_sec):
        if not self._lock.acquire(blocking=False):
            raise WorkerBusy('GPU worker already has an active operation')
        try:
            deadline=time.monotonic()+timeout_sec
            bundle=self.analyzer.analyze(video_path,job_id,timeout_sec,deadline=deadline)
            decision=finalize_unified(bundle,self.registry,release=self.release)
            windows={w['window_id']:w for w in bundle['report']['windows']}
            events=decision['supported']
            requests={str(i):[e['anchor_sec']] for i,e in enumerate(events)}
            evidence=extract_evidence_batch(video_path,requests,deadline=deadline) if requests else {}
            violations=[]
            for i,event in enumerate(events):
                ref=event['observation_refs'][0]
                observation=windows[ref['window_id']]['observations'][ref['observation_index']]
                frames=evidence[str(i)]
                require([f['timestamp_sec'] for f in frames]==requests[str(i)])
                violations.append({'type':event['type'],'timecode_sec':event['anchor_sec'],
                    'description':observation['description']+' '+event['description'],
                    'evidence_frames':frames})
            check_deadline(deadline)
            result=AnalysisResult.model_validate({'schema_version':'analysis.v1',
                'analysis_kind':'rules_evaluated','complete':True,'coverage_scope':'full_video',
                'rules_reviewed':True,'duration_sec':bundle['report']['video']['duration_sec'],
                'model_id':bundle['report']['model']['id'],
                'model_revision':bundle['report']['model']['revision'],
                'rules_version':decision['registry_version'],'pipeline_version':self.pipeline_version,
                'negative_policy_version':POLICY_VERSION,'verdict':decision['verdict'],
                'violations':violations,'rule_ids':[e['rule_id'] for e in events]})
            self.last_metrics=deepcopy(bundle['metrics'])
            self.last_manifest=deepcopy(bundle['report']['manifest'])
            return result.model_dump()
        finally:
            self._lock.release()


def model_identity(path):
    path=Path(path).resolve(strict=True)
    manifest_path=(path if path.is_dir() else path.parent)/'download_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    require(isinstance(manifest.get('repo_id'),str) and bool(manifest['repo_id'])
            and isinstance(manifest.get('revision'),str) and bool(manifest['revision']))
    return {'id':manifest['repo_id'],'revision':manifest['revision'],
            'download_manifest_sha256':fingerprint(manifest)}


def create_analyzer():
    """Load the real local components once, before worker readiness is announced."""
    from app.inference.qwen import QwenObserver
    from app.perception.detectors import RTDetr,OwlObjects
    from importlib.metadata import version
    vlm=os.environ['FORTE_MODEL_DIR']
    people=os.environ.get('FORTE_PEOPLE_MODEL_DIR','models/rtdetr')
    objects=os.environ.get('FORTE_OBJECT_MODEL_DIR','models/owlv2')
    quantization=os.environ.get('FORTE_MODEL_QUANTIZATION','nf4')
    backend=os.environ.get('FORTE_VLM_BACKEND','transformers')
    if backend not in ('transformers','vllm'):
        raise ValueError('FORTE_VLM_BACKEND must be transformers or vllm')
    if quantization not in (('nf4','fp16') if backend=='transformers' else ('fp8',)):
        raise ValueError('Unsupported quantization for selected backend')
    producers={'vlm':{**model_identity(vlm),'quantization':quantization},
               'people':{**model_identity(people),'adapter':'RTDetr','threshold':.3},
               'objects':{**model_identity(objects),'adapter':'OwlObjects','threshold':.1,
                          'queries':list(OwlObjects.QUERIES)}}
    for producer in producers.values():
        producer['runtime_versions']={name:version(name) for name in ('torch','transformers','bitsandbytes','av')}
    producers['vlm']['backend']=backend
    if backend=='vllm':
        from app.inference.vllm_observer import VllmObserver
        producers['vlm']['runtime_versions']['vllm']=version('vllm')
        observer=VllmObserver(vlm,max_new_tokens=1024)
    else:
        observer=QwenObserver(vlm,max_new_tokens=1024,quantize=quantization=='nf4')
    producers['vlm']['max_new_tokens']=observer.max_new_tokens
    return UnifiedAnalyzer(observer,RTDetr(people),OwlObjects(objects),producers=producers)


def create_processor():
    release_path=os.environ.get('FORTE_UNIFIED_RELEASE')
    if not release_path:
        raise RuntimeError('FORTE_UNIFIED_RELEASE requires an actual local acceptance record')
    release=json.loads(Path(release_path).read_text(encoding='utf-8'))
    registry=load_registry(os.environ.get('FORTE_RULES_PATH','config/rules.yaml'))
    return UnifiedProcessor(create_analyzer(),registry,release)
