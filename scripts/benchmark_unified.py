"""Run the real CV/targeted pipeline with one model load; print technical metrics only.

Detailed bundles stay local. This command reads no labels and activates no rules.
"""
import argparse
from collections import Counter
import contextlib
import gc
import json
import os
from pathlib import Path
import time

from app.jobs.unified import create_analyzer,finalize_unified
from app.inference.unified import PIPELINE_VERSION
from app.rules.engine import load_registry


def run_case(analyzer, video, output, timeout, index):
    output.mkdir(parents=True,exist_ok=False)
    metrics={'sample_index':index,'pipeline':PIPELINE_VERSION,'quality_measured':False,
             'public_service_ready':False,'stage':'analysis'}
    started=time.monotonic()
    before=analyzer.observer.metrics()['format_retries']
    repairs_before=getattr(analyzer.observer,'_null_geometry_normalizations',0)
    analyzer.observer.torch.cuda.reset_peak_memory_stats()
    def save_invalid_response(raw,metadata):
        directory=output/'model-errors'
        directory.mkdir(exist_ok=True)
        index=len(list(directory.glob('*.json')))
        (directory/f'{index:03}.json').write_text(json.dumps({'raw':raw,'metadata':metadata},ensure_ascii=False),encoding='utf-8')
    analyzer.observer._invalid_response_sink=save_invalid_response
    try:
        with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
            bundle=analyzer.analyze(video,f'local-evaluation-{index}',timeout_sec=timeout)
            (output/'bundle.json').write_text(json.dumps(bundle,ensure_ascii=False),encoding='utf-8')
            metrics.update(bundle['metrics'])
            metrics['stage']='rule_validation'
            decision=finalize_unified(bundle,load_registry())
            (output/'decision.json').write_text(json.dumps(decision,ensure_ascii=False),encoding='utf-8')
            facts=[v for p in bundle['assessments'] for v in p['facts'].values()]
            metrics.update({'complete':True,'stage':'complete',
                'facts':len(facts),'unknown_facts':sum(v is None for v in facts),
                'decisions':dict(Counter(p['decision'] for p in decision['candidates'])),
                'fact_supported_candidates':sum(p['reasons']==['rule_not_reviewed']
                    and p['decision']=='candidate' for p in decision['candidates']),
                'public_verdict_ready':decision['public_verdict_ready'],
                'model_id':analyzer.observer.model_id,'model_revision':analyzer.observer.model_revision,
                'people_frames':bundle['perception']['metrics']['people_frames'],
                'original_review_windows':bundle['perception']['metrics']['original_review_windows'],
                'object_frames':bundle['perception']['metrics']['object_frames']})
    except Exception as exc:
        metrics.update({'complete':False,'error_type':type(exc).__name__,
                        'elapsed_sec':round(time.monotonic()-started,3)})
        metrics.update(getattr(analyzer,'progress',{}))
    finally:
        analyzer.observer._invalid_response_sink=None
    gc.collect()
    analyzer.observer.torch.cuda.synchronize()
    stats=analyzer.observer.metrics()
    metrics.update(stats)
    metrics['validation_errors']=getattr(analyzer.observer,'_targeted_errors',[])
    metrics['null_geometry_normalizations']=getattr(analyzer.observer,'_null_geometry_normalizations',0)-repairs_before
    metrics['format_retries_this_task']=stats['format_retries']-before
    metrics['idle_allocated_mib']=round(analyzer.observer.torch.cuda.memory_allocated()/2**20,1)
    metrics['idle_reserved_mib']=round(analyzer.observer.torch.cuda.memory_reserved()/2**20,1)
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics),flush=True)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',required=True,nargs='+',type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--timeout',type=float,default=240.)
    parser.add_argument('--block-network',action='store_true')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    if args.block_network:
        import socket
        def denied(*args,**kwargs):
            raise RuntimeError('Runtime network access is disabled')
        socket.socket.connect=denied
        socket.socket.connect_ex=denied
        socket.create_connection=denied
    started=time.monotonic()
    try:
        with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
            analyzer=create_analyzer()
        summary={'load_sec':round(time.monotonic()-started,3),'cases':[],
                 'quality_measured':False,'public_service_ready':False}
        print(json.dumps({'models_loaded':True,'load_sec':summary['load_sec']}),flush=True)
    except Exception as exc:
        summary={'models_loaded':False,'error_type':type(exc).__name__}
        (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary),flush=True)
        return 1
    for index,video in enumerate(args.video):
        summary['cases'].append(run_case(analyzer,video,args.output_dir/f'{index:03}',args.timeout,index))
        (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return 0 if all(case['complete'] for case in summary['cases']) else 1


if __name__=='__main__':
    raise SystemExit(main())
