"""Evaluate files locally; stdout contains counts only."""
import argparse
import json
from pathlib import Path

from app.rules import evaluate, load_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observations', required=True)
    parser.add_argument('--assessments', required=True)
    parser.add_argument('--registry')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        report = json.loads(Path(args.observations).read_text(encoding='utf-8-sig'))
        packets = json.loads(Path(args.assessments).read_text(encoding='utf-8-sig'))
        result = evaluate(report, packets, load_registry(args.registry))
        output = Path(args.output).resolve()
        root = Path(__file__).resolve().parents[2]
        if not any(output.is_relative_to(root / folder) for folder in ('artifacts', 'results', 'data')):
            raise ValueError('Private output directory required')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    except (ValueError, KeyError, TypeError, OSError, IndexError, AttributeError):
        parser.exit(1, 'Rule evaluation failed; inspect local inputs. No scene content printed.\n')
    print(f"Evaluated {len(result['candidates'])} packets; supported: {len(result['supported'])}")


if __name__ == '__main__':
    main()
