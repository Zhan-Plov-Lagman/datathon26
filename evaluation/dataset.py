"""Private annotation manifest and immutable group-aware split generation."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def prepare(audit):
    if audit.get("issues"):
        raise ValueError("Resolve audit issues locally first")
    annotations = defaultdict(set)
    for video in audit["videos"]:
        annotations[video["sha256"]].add((video["label"], video["timestamp_sec"]))
    rows = []
    for video in audit["videos"]:
        rows.append({"file": video["file"], "sha256": video["sha256"],
                     "duration_sec": video["duration_sec"],
                     "organizer_label": video["label"],
                     "organizer_timestamp_raw": video["timestamp_raw"],
                     "organizer_timestamp_sec": video["timestamp_sec"],
                     "scene_group": None, "reviewed": False, "label": None,
                     "ambiguous": len(annotations[video["sha256"]]) > 1,
                     "events": [], "hard_negative_tags": []})
    return {"schema_version": "annotations.v1", "scene_grouping_reviewed": False,
            "records": rows}


def select_records(manifest, split, partition):
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if digest != split["manifest_sha256"]:
        raise ValueError("Manifest changed since split freeze; review independence")
    files = {name for group in split["groups"][partition] for name in group}
    grouped = defaultdict(list)
    for row in manifest["records"]:
        if row["file"] in files:
            grouped[row["sha256"]].append(row)
    records = []
    for sha, rows in sorted(grouped.items()):
        row = dict(min(rows, key=lambda r: r["file"]))
        # Never silently select one of two incompatible annotations.
        if any(r['events'] != row['events'] or r['label'] != row['label'] for r in rows):
            row['ambiguous'] = True
        row['ambiguous'] = row['ambiguous'] or any(r['ambiguous'] for r in rows)
        row['reviewed'] = all(r['reviewed'] for r in rows)
        row['id'] = row['file']
        records.append(row)
    return {"records": records, "partition": partition, "manifest_sha256": digest}


def make_split(manifest, seed=26):
    rows = manifest["records"]
    if not rows or manifest.get("scene_grouping_reviewed") is not True:
        raise ValueError("Human review of independent scenes is required")
    names = [r["file"] for r in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate manifest record")
    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = {}
    for i, row in enumerate(rows):
        if not row.get("scene_group") or not row.get("sha256") or type(row.get("label")) is not bool:
            raise ValueError("Complete scene groups and binary labels locally")
        for key in (("hash", row["sha256"]), ("scene", row["scene_group"])):
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    if len(groups) < 3:
        raise ValueError("At least three independent groups required")
    # Stable ordering before seeded shuffle: input row order cannot alter split.
    units = sorted(groups.values(), key=lambda group: min(r["file"] for r in group))
    random.Random(seed).shuffle(units)
    units.sort(key=lambda group: len({r["sha256"] for r in group}), reverse=True)
    ratios = {"development": .6, "tuning": .2, "holdout": .2}
    unique = {r["sha256"]: r["label"] for r in rows}
    totals = Counter(unique.values())
    assigned = {k: [] for k in ratios}
    counts = {k: Counter() for k in ratios}
    for index, group in enumerate(units):
        labels_by_hash = defaultdict(set)
        for row in group:
            labels_by_hash[row["sha256"]].add(row["label"])
        if any(len(v) != 1 for v in labels_by_hash.values()):
            raise ValueError("Conflicting labels in exact duplicate group")
        weight = Counter(next(iter(v)) for v in labels_by_hash.values())
        empty = [k for k in ratios if not assigned[k]]
        choices = empty if len(units) - index == len(empty) else list(ratios)
        def cost(target):
            return sum((counts[k][label] + (weight[label] if k == target else 0)
                        - ratios[k] * totals[label]) ** 2
                       for k in ratios for label in totals)
        target = min(choices, key=cost)
        counts[target].update(weight)
        assigned[target].append(sorted(r["file"] for r in group))
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"schema_version": "splits.v1", "seed": seed, "manifest_sha256": digest,
            "independence": "scene_reviewed_before_tuning", "groups": assigned,
            "unique_counts": {k: sum(v.values()) for k, v in counts.items()}}


def write_private(path, content):
    path = Path(path).resolve()
    root = Path(__file__).resolve().parents[1]
    if not any(path.is_relative_to(root / folder) for folder in ("data", "artifacts", "results")):
        raise ValueError("Private output must be in data/, artifacts/ or results/")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
