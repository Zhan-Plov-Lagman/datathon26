"""Setup only. Downloads public weights; never reads or uploads bank data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", default="models/Qwen3-VL-2B-Instruct")
    args = parser.parse_args()
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(args.repo_id, revision=args.revision, token=False)
    output = Path(args.output_dir).resolve()
    snapshot_download(args.repo_id, revision=info.sha, local_dir=output, token=False, max_workers=2,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE*"])
    (output / "download_manifest.json").write_text(json.dumps(
        {"repo_id": args.repo_id, "revision": info.sha}, indent=2), encoding="utf-8")
    print(json.dumps({"downloaded": True, "repo_id": args.repo_id, "revision": info.sha}))


if __name__ == "__main__":
    main()
