"""Private note-assisted development review. Hints are NEVER ground truth.

Does not run inference or feed annotations to the model. Console is counts only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


TOKEN = re.compile(r'(?<!\w)[xх][-– ]?(\d+)(?!\d)', re.I)


def group_notes(text, videos):
    by_number = {}
    for path in videos:
        match = TOKEN.fullmatch(path.stem)
        if match:
            number = int(match[1])
            if number in by_number:
                raise ValueError('Ambiguous normalized video identifiers')
            by_number[number] = path
    blocks = {number: [] for number in by_number}
    active = None
    for line in text.splitlines():
        matches = list(TOKEN.finditer(line))
        # Conservative short heading detection; uncertain fragments stay in notes.
        if len(line) <= 120 and len(matches) == 1 and int(matches[0][1]) in by_number:
            active = int(matches[0][1])
        if active is not None:
            blocks[active].append(line)
    return [(by_number[k], '\n'.join(lines)) for k, lines in blocks.items() if lines]


def suggest_cases(grouped):
    rows = []
    seen = set()
    for path, note in grouped:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        lower = note.lower()
        departure = len(re.findall(r'покин|покид|уходит|уход|отош|отход', lower))
        motion = len(re.findall(r'вста[её]|поворач|наклон|возвращ|садится', lower))
        rows.append({'video_path': str(path.resolve()), 'video_url': path.resolve().as_uri(),
                     'sha256': digest, 'notes': note, '_departure': departure, '_motion': motion})
    if not rows:
        raise ValueError('No note blocks matched local videos')
    positive = max(rows, key=lambda r:r['_departure'])
    others = [r for r in rows if r is not positive]
    comparison = max(others, key=lambda r:(r['_motion'], -r['_departure'])) if others else None
    chosen = [(positive, 'possible_departure' if positive['_departure'] else 'unclassified')]
    if comparison:
        chosen.append((comparison, 'comparison_requires_review'))
    result = []
    for index, (row, hint) in enumerate(chosen):
        result.append({k:v for k,v in row.items() if not k.startswith('_')} | {
            'id': f'development-review-{index+1:02}', 'selection_hint': hint,
            'label': None, 'reviewed': False, 'events': [], 'reviewer': '',
            'split': 'development', 'independent_validation': False})
    return result, len(rows)


def reviewed_records(data):
    """Explicit human decisions only; never turn a search hint into a label."""
    rows = data['cases']
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicated review cases')
    for row in rows:
        if row.get('reviewed') is not True or type(row.get('label')) is not bool:
            raise ValueError('Human binary review required; unknown is not negative')
        if not isinstance(row.get('reviewer'), str) or not row['reviewer'].strip():
            raise ValueError('Reviewer required')
        if row.get('split') != 'development' or row.get('independent_validation') is not False:
            raise ValueError('These note-selected cases are development only')
        events = row.get('events')
        if not isinstance(events, list) or bool(events) != row['label']:
            raise ValueError('Label and event list disagree')
        if any(e.get('type') != 'cashier_leaves_service' or
               type(e.get('timecode_sec')) not in (int,float) or
               not 0 <= e['timecode_sec'] < float('inf') for e in events):
            raise ValueError('Finite nonnegative event time required')
        path = Path(row['video_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Source changed since review preparation')
    if len({r['sha256'] for r in rows}) != len(rows):
        raise ValueError('Duplicate footage is not an independent case')
    return [{k:r[k] for k in ('id','sha256','label','reviewed','events')} | {'ambiguous':False}
            for r in rows]


def private_output(path):
    path = Path(path).resolve()
    # Protect accidental Git publication even when run from another worktree.
    if not any(part in ('artifacts','data','results') for part in path.parts):
        raise ValueError('Output must be inside a private artifacts/data/results directory')
    if path.exists():
        raise ValueError('Output already exists')
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--notes')
    parser.add_argument('--video-dir')
    parser.add_argument('--reviewed-input')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = private_output(args.output)
    if args.reviewed_input:
        records = reviewed_records(json.loads(Path(args.reviewed_input).read_text(encoding='utf-8')))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({'records':records},indent=2),encoding='utf-8')
        print(json.dumps({'reviewed_cases':len(records)}))
        return
    if not args.notes or not args.video_dir:
        parser.error('--notes and --video-dir required')
    videos = [p for p in Path(args.video_dir).iterdir() if p.suffix.lower()=='.mov']
    grouped = group_notes(Path(args.notes).read_text(encoding='utf-8-sig'), videos)
    cases, unique = suggest_cases(grouped)
    output.mkdir(parents=True)
    data = {'schema_version':'development_review.v1','cases':cases}
    (output/'cases.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    template = Path(__file__).with_name('note_review.html').read_text(encoding='utf-8')
    payload = json.dumps(data,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    (output/'review.html').write_text(template.replace('__LOCAL_DATA__',payload),encoding='utf-8')
    print(json.dumps({'matched_note_blocks':len(grouped),'unique_videos':unique,'selected_cases':len(cases),
                      'reviewed_cases':0,'quality_metrics_available':False}))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        # Never expose source lines, paths from traceback or annotation content.
        print(json.dumps({'complete':False,'error_type':type(exc).__name__}))
        raise SystemExit(1) from None
