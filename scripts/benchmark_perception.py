"""Local detector benchmark. Only aggregate timing/coverage leaves stdout."""
import argparse
import contextlib
import json
import os
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path)
    parser.add_argument('--backend',choices=['rtdetr','yolo','owlv2'],required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--fps',type=float,default=3)
    parser.add_argument('--timeout',type=float,default=240)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    from app.perception.detectors import RTDetr,YoloPose,OwlObjects
    from app.perception.stream import iter_batches
    from app.video.sampling import probe_video
    import torch
    result={'schema_version':'perception_benchmark.v1','backend':args.backend,'input_fps':args.fps,
            'frames':0,'detections':0,'complete':False,'quality_measured':False}
    started=time.monotonic()
    with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
        try:
            detector={'rtdetr':RTDetr,'yolo':YoloPose,'owlv2':OwlObjects}[args.backend](args.weights)
            result['load_sec']=round(time.monotonic()-started,3)
            torch.cuda.reset_peak_memory_stats()
            t=time.monotonic(); deadline=t+args.timeout
            info=probe_video(args.video,deadline)
            result['duration_sec']=info.duration_sec
            first,last=None,None
            for batch in iter_batches(args.video,fps=args.fps,deadline=deadline,info=info):
                detections=detector(batch,deadline)
                if len(detections)!=len(batch):
                    raise ValueError('Incomplete detector batch')
                first=batch[0].timestamp_sec if first is None else first
                last=batch[-1].timestamp_sec
                result['frames']+=len(batch)
                result['detections']+=sum(map(len,detections))
            result.update(complete=True,processing_sec=round(time.monotonic()-t,3),
                          first_pts=first,last_pts=last)
        except Exception as exc:
            result['error_type']=type(exc).__name__
        finally:
            result['wall_sec']=round(time.monotonic()-started,3)
            result['peak_reserved_mib']=round(torch.cuda.max_memory_reserved()/2**20,2)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
    return 0 if result['complete'] else 1


if __name__=='__main__':
    raise SystemExit(main())
