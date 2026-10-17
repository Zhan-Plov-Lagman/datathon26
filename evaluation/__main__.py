import argparse
import json
from pathlib import Path

from .dataset import make_split, prepare, select_records, write_private
from .metrics import score


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description="Local-only role B utilities; no scene content in console")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "split", "select", "score"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--input", required=True)
        cmd.add_argument("--output", required=True)
        if name == "score":
            cmd.add_argument("--predictions", required=True)
        if name == "split":
            cmd.add_argument("--seed", type=int, default=26)
        if name == "select":
            cmd.add_argument("--splits", required=True)
            cmd.add_argument("--partition", choices=["development", "tuning", "holdout"], required=True)
    args = parser.parse_args()
    try:
        body = read(args.input)
        if args.command == "prepare":
            result = prepare(body)
        elif args.command == "split":
            result = make_split(body, args.seed)
        elif args.command == "select":
            result = select_records(body, read(args.splits), args.partition)
        else:
            result = score(body["records"], read(args.predictions)["predictions"])
        write_private(args.output, result)
    except (ValueError, KeyError, TypeError, OSError, IndexError):
        parser.exit(1, "Local input/output validation failed; inspect files locally. No data printed.\n")
    print("Completed; result saved locally. No scene content printed.")


if __name__ == "__main__":
    main()
