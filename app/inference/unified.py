"""CV proposals -> shared targeted VLM review -> grounded fact packets.

This producer is suitable for local evaluation before a detector release exists.
It never supplies a public verdict and never reads annotation files.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from threading import Lock
import time

from app.inference.targeted import FACTS, PROMPT, VERSION, review
from app.inference.worker import WorkerBusy
from app.perception.scan import scan_video
from app.perception.temporal import iou
from app.video.sampling import check_deadline, probe_video, sample_frames


PIPELINE_VERSION = 'cv-targeted.v1.3'
PREPROCESSING = {'cv_fps':3., 'object_period_sec':5., 'cv_edge':960,
                 'review_frames':12, 'review_span_sec':20., 'review_edge':768,
                 'refine_sec':4., 'refine_frames':6}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()


def _frames(sampled, targets):
    unique = {sampled[t].timestamp_sec:sampled[t] for t in targets}
    return [unique[t] for t in sorted(unique)]


def _targets(start, end, count):
    return [round(start+(end-start)*i/(count-1),6) for i in range(count)]


def _actor_at(scan, box, timestamp):
    if box is None:
        return None
    sample = min(scan['samples'],key=lambda s:abs(s['pts']-timestamp))
    if abs(sample['pts']-timestamp) > .5:
        return None
    matches = [(iou(box,a['box']),a['id']) for a in sample['actors']]
    matches = sorted((x for x in matches if x[0]>=.3),reverse=True)
    # Do not merge two nearby actors on a weak/ambiguous spatial match.
    if not matches or (len(matches)>1 and matches[0][0]-matches[1][0]<.15):
        return None
    return matches[0][1]


class UnifiedAnalyzer:
    def __init__(self, observer, people, objects, *, producers):
        self.observer,self.people,self.objects = observer,people,objects
        self.manifest = {
            'schema_version':'producer_manifest.v1', 'pipeline_version':PIPELINE_VERSION,
            'producers':deepcopy(producers), 'preprocessing':deepcopy(PREPROCESSING),
            'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
            'fact_version':VERSION,
            'implementation_sha256':{name:hashlib.sha256((Path(__file__).resolve().parents[2]/name).read_bytes()).hexdigest()
                for name in ('app/inference/targeted.py','app/inference/unified.py','app/inference/qwen.py',
                    'app/perception/scan.py','app/perception/temporal.py','app/perception/detectors.py',
                    'app/perception/stream.py','app/video/sampling.py','app/jobs/unified.py',
                    'app/rules/engine.py','app/evidence/frames.py')},
        }
        self._lock = Lock()

    def analyze(self, video_path, job_id, timeout_sec=240., *, deadline=None):
        if not self._lock.acquire(blocking=False):
            raise WorkerBusy('GPU worker already has an active operation')
        try:
            if not isinstance(job_id,str) or not job_id or not math.isfinite(timeout_sec) or timeout_sec<=0:
                raise ValueError('Invalid job/budget')
            started = time.monotonic()
            limit = started+timeout_sec
            deadline = limit if deadline is None else min(deadline,limit)
            check_deadline(deadline)
            return self._analyze(video_path,job_id,deadline,started)
        finally:
            self._lock.release()

    def _analyze(self, path, job_id, deadline, started):
        self.progress={'stage':'scan','completed_review_calls':0,'completed_refinement_calls':0}
        self.observer._targeted_errors=[]
        cfg = self.manifest['preprocessing']
        scan = scan_video(path,self.people,self.objects,job_id=job_id,
            timeout_sec=max(.001,deadline-time.monotonic()),deadline=deadline,
            fps=cfg['cv_fps'],object_period_sec=cfg['object_period_sec'],max_long_edge=cfg['cv_edge'],
            review_span_sec=cfg['review_span_sec'])
        info = probe_video(path,deadline)
        planned = scan['review_windows']
        self.progress.update({'stage':'review','scan_sec':scan['metrics']['elapsed_sec'],
            'people_frames':scan['metrics']['people_frames'],'object_frames':scan['metrics']['object_frames'],
            'planned_review_calls':len(planned),
            'original_review_windows':scan['metrics']['original_review_windows']})
        if not planned:
            raise ValueError('No targeted review plan')
        target_lists = [_targets(w['start_sec'],w['end_sec'],cfg['review_frames']) for w in planned]
        decode_started = time.monotonic()
        sampled = sample_frames(path,[t for row in target_lists for t in row],info,
                                max_long_edge=cfg['review_edge'],deadline=deadline)
        decode_sec = time.monotonic()-decode_started
        windows, review_runs, pending = [],[],[]
        vlm_started = time.monotonic()
        for index, targets in enumerate(target_lists):
            frames = _frames(sampled,targets)
            result = review(self.observer,frames,deadline)
            pts = [f.timestamp_sec for f in frames]
            windows.append(self._window(index,pts,result,'proposal'))
            review_runs.append({'window_id':index,'frame_timestamps_sec':pts,
                                'attempts':result['attempts'],'status':'complete'})
            self.progress['completed_review_calls']=len(review_runs)
            # Every proposal is inspected densely, including incomplete fact packets.
            for event in result['events']:
                anchor = pts[event['anchor_frame']]
                start = max(0.,anchor-cfg['refine_sec']/2)
                end = min(info.duration_sec,start+cfg['refine_sec'])
                start = max(0.,end-cfg['refine_sec'])
                # Duplicate intervals share a call, not invented inter-window identities.
                key = (round(start,6),round(end,6))
                if key not in pending:
                    pending.append(key)
        del sampled
        overview_sec = time.monotonic()-vlm_started
        dense_targets = [_targets(a,b,cfg['refine_frames']) for a,b in pending]
        dense_started = time.monotonic()
        self.progress.update({'stage':'refinement','planned_refinement_calls':len(pending)})
        sampled = sample_frames(path,[t for row in dense_targets for t in row],info,
                                max_long_edge=cfg['review_edge'],deadline=deadline)
        assessments, episodes, refine_runs = [],[],[]
        for targets in dense_targets:
            frames = _frames(sampled,targets)
            result = review(self.observer,frames,deadline)
            pts = [f.timestamp_sec for f in frames]
            wid = len(windows)
            window = self._window(wid,pts,result,'refinement')
            windows.append(window)
            refine_runs.append({'window_id':wid,'frame_timestamps_sec':pts,
                               'attempts':result['attempts'],'status':'complete'})
            self.progress['completed_refinement_calls']=len(refine_runs)
            for index,event in enumerate(result['events']):
                eid = f'w{wid}-e{index}'
                ref = {'window_id':wid,'observation_index':index}
                facts = {name:item['value'] for name,item in event['facts'].items()}
                refs=[ref]
                sources={}
                for name,item in event['facts'].items():
                    if item['value'] is None:
                        sources[name]={'origin':'unknown','observation_refs':[],
                            'frame_timestamps_sec':[],'reason':'not_observable'}
                        continue
                    indices=item['frames']
                    feature_ref={'window_id':wid,'observation_index':len(window['observations'])}
                    window['observations'].append({'start_frame':indices[0],'end_frame':indices[-1],
                        'start_sec':pts[indices[0]],'end_sec':pts[indices[-1]],
                        'feature_key':name,'value':item['value'],
                        'description':f'{name}={str(item["value"]).lower()}'})
                    refs.append(feature_ref)
                    sources[name]={'origin':'visual','observation_refs':[feature_ref],
                        'frame_timestamps_sec':[pts[i] for i in indices],
                        'reason':'visible_in_referenced_frames'}
                assessments.append({'rule_id':event['rule_id'],'event_id':eid,
                    'facts':facts,'observation_refs':refs, 'anchor_sec':pts[event['anchor_frame']],
                    'refinement_complete':True,'full_sequence_visible':False})
                episodes.append({'rule_id':event['rule_id'],'event_id':eid,
                    'actor_track_id':_actor_at(scan,event['actor_box'],pts[event['anchor_frame']]),
                    'actor_box':event['actor_box'], 'start_sec':pts[event['start_frame']],
                    'end_sec':pts[event['end_frame']],
                    'fact_sources':sources})
        check_deadline(deadline)
        report = {'schema_version':'observations.v2','analysis_kind':'observations_only',
            'job_id':job_id,'complete':True,'coverage_scope':'cv_full_video_vlm_targeted',
            'video':{'duration_sec':info.duration_sec},
            'model':{'id':self.observer.model_id,'revision':self.observer.model_revision},
            'windows':windows,'manifest':deepcopy(self.manifest),
            'limitations':['CV and VLM frames are sampled, not continuous visibility.',
                'Actor tracks and roles are visual hypotheses; no biometric identity.',
                'Only departure and personal-item facts are implemented at this stage.']}
        receipt = {'schema_version':'detector_execution.v2','job_id':job_id,
            'manifest_sha256':fingerprint(self.manifest),'supported_rule_ids':list(FACTS),
            'cv':{'status':'complete','coverage_scope':'full_video_sampled',
                'people_frame_timestamps_sec':[s['pts'] for s in scan['samples']],
                'object_frame_timestamps_sec':[s['pts'] for s in scan['object_samples']]},
            'planned_review_windows':deepcopy(planned),'review_runs':review_runs,
            'planned_refinement_intervals':[list(pair) for pair in pending],
            'refinement_runs':refine_runs}
        return {'schema_version':'assessment_bundle.v2','complete':True,'report':report,
            'perception':scan,'assessments':assessments,
            'diagnostics':{'search':receipt,'episodes':episodes},
            'metrics':{'scan_sec':scan['metrics']['elapsed_sec'],'decode_sec':round(decode_sec,3),
                'overview_sec':round(overview_sec,3),'refinement_sec':round(time.monotonic()-dense_started,3),
                'total_sec':round(time.monotonic()-started,3),'review_calls':len(review_runs),
                'refinement_calls':len(refine_runs),'candidates':len(assessments)}}

    @staticmethod
    def _window(wid, pts, result, stage):
        return {'window_id':wid,'stage':stage,'start_sec':pts[0],'end_sec':pts[-1],
            'frame_timestamps_sec':pts,'visibility_limits':[],
            'observations':[{'start_frame':e['start_frame'],'end_frame':e['end_frame'],
                'start_sec':pts[e['start_frame']],'end_sec':pts[e['end_frame']],
                'anchor_sec':pts[e['anchor_frame']],'rule_id':e['rule_id'],
                'description':e['description']} for e in result['events']]}
