"""Copy selected peer source files from fetched Git refs into an ignored test snapshot."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b-ref", default="origin/feature/rules-eval")
    parser.add_argument("--c-ref", default="origin/feature/api-ui")
    parser.add_argument("--output", default=".cache/peer-contracts")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    git = ["git", "-c", f"safe.directory={root.as_posix()}"]

    def read(*arguments):
        return subprocess.run([*git, *arguments], cwd=root, check=True, capture_output=True).stdout

    versions = {"B": read("rev-parse", "--verify", args.b_ref + "^{commit}").decode().strip(),
                "C": read("rev-parse", "--verify", args.c_ref + "^{commit}").decode().strip()}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = {"B": ["app/rules/__init__.py", "app/rules/engine.py", "config/rules.yaml"],
             "C": ["app/api/__init__.py", "app/api/schema.py", "app/jobs/__init__.py", "app/jobs/pipeline.py"]}
    for role, relative in (("B", "app/rules/policy.py"), ("C", "app/jobs/detection.py"),
                           ("C", "app/jobs/provenance.py"), ("C", "app/jobs/evaluator.py")):
        present = read("ls-tree", "--name-only", versions[role], "--", relative).decode().strip()
        if present:
            files[role].append(relative)
    for role, paths in files.items():
        for relative in paths:
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(read("show", f"{versions[role]}:{relative}"))
    (output / "revisions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    print(json.dumps({"snapshot": str(output), "revisions": versions}))


if __name__ == "__main__":
    main()
