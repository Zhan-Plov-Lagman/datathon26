"""Explicit online setup; runtime adapters never download. Only public model files."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

MODELS = {
    'rtdetr': ('PekingU/rtdetr_v2_r50vd','282494075698cab9faa1096ae26856890030c817',None),
    'owlv2': ('google/owlv2-base-patch16-ensemble','cfd3195ba4ea9592eec887ded089f4c08eff231d',None),
    'dinov2': ('facebook/dinov2-base','f9e44c814b77203eaa57a6bdbbd535f21ede1415',None),
    'yolo11m-pose': ('Ultralytics/YOLO11','8b8ac7d1fae7468f85dbf89670dd66f41f485aab','yolo11m-pose.pt'),
    'qwen4b': ('Qwen/Qwen3-VL-4B-Instruct','ebb281ec70b05090aa6165b016eac8ec08e71b17',None),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models',nargs='+',choices=list(MODELS))
    parser.add_argument('--root',type=Path,default=Path('models'))
    args=parser.parse_args()
    from huggingface_hub import HfApi,snapshot_download
    api=HfApi()
    args.root.mkdir(parents=True,exist_ok=True)
    for name in args.models:
        repo,revision,single=MODELS[name]
        info=api.model_info(repo,revision=revision,files_metadata=True)
        allow=[f.rfilename for f in info.siblings if
            (single and f.rfilename==single) or
            (not single and f.rfilename.endswith(('.safetensors','.json','.txt','.model','.jinja')))
            or f.rfilename.upper() in ('LICENSE','LICENSE.MD','NOTICE','README.MD')]
        destination=args.root/name
        needed=sum(f.size or 0 for f in info.siblings if f.rfilename in allow and not (destination/f.rfilename).is_file())
        if shutil.disk_usage(args.root).free < needed + 2*1024**3:
            print(json.dumps({'model':name,'status':'insufficient_disk','download_bytes':needed}))
            return 2
        snapshot_download(repo,revision=revision,local_dir=destination,allow_patterns=allow,max_workers=2)
        files=[]
        for relative in allow:
            path=destination/relative
            if not path.is_file():
                raise RuntimeError('Model snapshot is incomplete')
            with path.open('rb') as f:
                digest=hashlib.file_digest(f,'sha256').hexdigest()
            files.append({'path':relative,'sha256':digest,'bytes':path.stat().st_size})
        (destination/'download_manifest.json').write_text(json.dumps({
            'repo_id':repo,'revision':revision,'files':files},indent=2),encoding='utf-8')
        print(json.dumps({'model':name,'status':'downloaded','files':len(files)}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
