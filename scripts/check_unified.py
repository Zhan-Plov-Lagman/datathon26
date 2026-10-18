"""Technical preflight only; never announce analysis readiness from installed packages."""
import importlib
import json
import shutil
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'local-paths.json').read_text(encoding='utf-8'))
    checks={}
    for name in ('torch','transformers','av','fastapi','pydantic','httpx','ultralytics','sklearn'):
        try:
            module=importlib.import_module(name)
            checks[name]={'import':True,'version':getattr(module,'__version__','unknown')}
        except Exception as exc:
            checks[name]={'import':False,'error_type':type(exc).__name__}
    checks['qwen2b']=(Path(config['qwen2b'])/'config.json').is_file()
    checks['ffprobe']=shutil.which('ffprobe') is not None
    checks['ffmpeg']=shutil.which('ffmpeg') is not None
    if checks['torch']['import']:
        import torch
        checks['cuda']=torch.cuda.is_available()
        if checks['cuda']:
            checks['vram_mib']=torch.cuda.get_device_properties(0).total_memory//2**20
    checks['disk_free_gib']=round(shutil.disk_usage(root).free/2**30,2)
    checks['analysis_quality_verified']=False
    print(json.dumps(checks,indent=2))
    return 0 if all(v['import'] for v in checks.values() if isinstance(v,dict)) and checks.get('cuda') and checks['qwen2b'] and checks['ffmpeg'] and checks['ffprobe'] else 1


if __name__=='__main__':
    raise SystemExit(main())
