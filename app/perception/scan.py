"""One shared streaming CV pass. Produces proposals, not a jury result."""
from dataclasses import asdict
import math
import time

from app.video.sampling import probe_video, check_deadline
from app.perception.stream import iter_batches
from app.perception.temporal import Tracker, motion_signal, plan_reviews


def scan_video(path, people, objects, *, job_id, timeout_sec=240., deadline=None,
               fps=3., object_period_sec=5., max_long_edge=960):
    if not all(math.isfinite(x) and x>0 for x in (timeout_sec, object_period_sec)):
        raise ValueError('Positive processing budgets required')
    deadline=min(deadline or float('inf'),time.monotonic()+timeout_sec)
    started=time.monotonic()
    info=probe_video(path,deadline)
    tracker=Tracker(max_gap_sec=max(1.,2/fps))
    samples,signals,object_samples=[],[],[]
    next_object=0.
    last_motion={}
    last_count=None
    for batch in iter_batches(path,info=info,fps=fps,max_long_edge=max_long_edge,deadline=deadline):
        detections=people(batch,deadline)
        if len(detections)!=len(batch):
            raise ValueError('Detector skipped input frames')
        for frame,detected in zip(batch,detections,strict=True):
            timestamp=frame.timestamp_sec
            tracks=tracker.update(timestamp,detected)
            if last_count is not None and len(tracks)!=last_count:
                signals.append((timestamp,'people_count_change',None))
            last_count=len(tracks)
            for track in tracks:
                if motion_signal(track) and timestamp-last_motion.get(track.id,-100.)>=5:
                    signals.append((timestamp,'actor_motion',track.id))
                    last_motion[track.id]=timestamp
            samples.append({'pts':timestamp,'actors':[{'id':t.id,'box':t.box} for t in tracks]})
            if timestamp+1e-6>=next_object:
                result=objects([frame],deadline)
                if len(result)!=1:
                    raise ValueError('Object detector skipped input frame')
                object_samples.append({'pts':timestamp,'detections':[asdict(d) for d in result[0]]})
                if any(d.label in ('banknotes','a stack of paper money','a handbag','a coat','an open cabinet door')
                       for d in result[0]):
                    signals.append((timestamp,'object_state',None))
                next_object=timestamp+object_period_sec
    check_deadline(deadline)
    if not samples or not object_samples:
        raise ValueError('Empty CV execution')
    windows=plan_reviews(info.duration_sec,signals)
    return {
        'schema_version':'perception.v1','job_id':job_id,'complete':True,
        'coverage_scope':'full_video_sampled','duration_sec':info.duration_sec,
        'sample_fps':fps,'object_period_sec':object_period_sec,'samples':samples,
        'object_samples':object_samples,
        'review_windows':[{'start_sec':w.start_sec,'end_sec':w.end_sec,
                           'reasons':sorted(w.reasons),'actor_ids':sorted(w.actor_ids)} for w in windows],
        'metrics':{'elapsed_sec':round(time.monotonic()-started,3),
                   'people_frames':len(samples),'object_frames':len(object_samples),
                   'review_windows':len(windows)},
        'limitations':['Sampling does not establish continuous visibility.',
                       'Track IDs are local hypotheses; no cashier identity is assigned.'],
    }
