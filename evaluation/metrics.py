"""Aggregate metrics only. Failed/missing predictions remain visible."""
import math
from collections import Counter, defaultdict


def ratio(a, b):
    return a / b if b else None


def match_events(truth, predictions, tolerance=5, typed=True):
    """Maximum cardinality bipartite matching; predictions cannot match twice."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid tolerance")
    for event in truth + predictions:
        t = event.get("timecode_sec")
        if type(t) not in (int, float) or not math.isfinite(t) or t < 0 or not event.get("type"):
            raise ValueError("Invalid event")
    edges = [[j for j, p in sorted(enumerate(predictions),
              key=lambda item: abs(t["timecode_sec"] - item[1]["timecode_sec"]))
              if abs(t["timecode_sec"] - p["timecode_sec"]) <= tolerance
              and (not typed or t["type"] == p["type"])] for t in truth]
    matched = {}
    def visit(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matched or visit(matched[j], seen):
                matched[j] = i
                return True
        return False
    for i in range(len(truth)):
        visit(i, set())
    return sorted((i, j) for j, i in matched.items())


def score(records, predictions, tolerance=5):
    """Records are one reviewed representative per exact hash in a fixed split.

    Input predictions: id, status, verdict, violations, latency_sec. The caller
    must use actual runtime outputs; identifiers are only used to join for scoring.
    """
    ids = [r["id"] for r in records]
    pids = [p["id"] for p in predictions]
    if len(ids) != len(set(ids)) or len(pids) != len(set(pids)) or not set(pids) <= set(ids):
        raise ValueError("Duplicate or unknown evaluation ids")
    hashes = [r["sha256"] for r in records]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Evaluate one representative per exact duplicate group")
    by_id = {p["id"]: p for p in predictions}
    counts = Counter(tp=0, tn=0, fp=0, fn=0, failed=0, missing=0, ambiguous=0)
    event_counts = Counter(tp=0, fp=0, fn=0)
    per_class = defaultdict(lambda: Counter(tp=0, fp=0, fn=0))
    errors, latencies = [], []
    localization_hits = Counter()
    classification_correct = classification_matched = 0
    for record in records:
        if type(record.get("label")) is not bool or record.get("reviewed") is not True:
            raise ValueError("Reviewed binary ground truth required")
        if record.get("ambiguous"):
            counts["ambiguous"] += 1
            continue
        truth = record["events"]
        if bool(truth) != record["label"]:
            raise ValueError("Ground truth events and verdict disagree")
        # Validate even when the prediction is missing.
        match_events(truth, [], tolerance)
        prediction = by_id.get(record["id"])
        if prediction is None:
            counts["missing"] += 1
            event_counts["fn"] += len(truth)
            for event in truth:
                per_class[event['type']]['fn'] += 1
            continue
        status = prediction.get("status")
        if status not in ("done", "failed"):
            raise ValueError("Only terminal predictions can be scored")
        latency = prediction.get("latency_sec")
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            raise ValueError("Finite end-to-end latency required")
        latencies.append(latency)
        if status == "failed":
            counts["failed"] += 1
            event_counts["fn"] += len(truth)
            for event in truth:
                per_class[event['type']]['fn'] += 1
            continue
        pred = prediction["violations"]
        verdict = prediction.get("verdict")
        if verdict not in ("violation", "no_violation") or bool(pred) != (verdict == "violation"):
            raise ValueError("Inconsistent public verdict")
        actual, detected = record["label"], bool(pred)
        counts["tp" if actual and detected else "fn" if actual else "fp" if detected else "tn"] += 1
        pairs = match_events(truth, pred, tolerance)
        event_counts.update(tp=len(pairs), fp=len(pred) - len(pairs), fn=len(truth) - len(pairs))
        matched_truth, matched_pred = {i for i, _ in pairs}, {j for _, j in pairs}
        for i, event in enumerate(truth):
            per_class[event['type']]['tp' if i in matched_truth else 'fn'] += 1
        for j, event in enumerate(pred):
            if j not in matched_pred:
                per_class[event['type']]['fp'] += 1
        untyped = match_events(truth, pred, tolerance, typed=False)
        classification_matched += len(untyped)
        classification_correct += sum(truth[i]["type"] == pred[j]["type"] for i, j in untyped)
        errors.extend(abs(truth[i]["timecode_sec"] - pred[j]["timecode_sec"]) for i, j in pairs)
        for seconds in (1, 2, 5):
            localization_hits[seconds] += len(match_events(truth, pred, seconds))
    for metrics in (counts, event_counts, *per_class.values()):
        metrics["precision"] = ratio(metrics["tp"], metrics["tp"] + metrics["fp"])
        metrics["recall"] = ratio(metrics["tp"], metrics["tp"] + metrics["fn"])
        metrics["f1"] = ratio(2 * metrics["tp"], 2 * metrics["tp"] + metrics["fp"] + metrics["fn"])
    n = len(records) - counts["ambiguous"]
    gt_events = sum(len(r["events"]) for r in records if not r.get("ambiguous"))
    sorted_latency = sorted(latencies)
    return {"videos": dict(counts), "video_metrics_scope": "successful_nonambiguous_only",
            "total": len(records), "completion_rate": ratio(n - counts["missing"] - counts["failed"], n),
            "events": dict(event_counts), "event_tolerance_sec": tolerance,
            "per_class": {key: dict(value) for key, value in sorted(per_class.items())},
            "localization_hits": {str(s): {"count": localization_hits[s], "fraction": ratio(localization_hits[s], gt_events)} for s in (1, 2, 5)},
            "matched_time_mae_sec": ratio(sum(errors), len(errors)),
            "classification_accuracy_on_temporal_matches": ratio(classification_correct, classification_matched),
            "latency_count": len(latencies), "latency_max_sec": max(latencies, default=None),
            "latency_p95_sec": sorted_latency[math.ceil(.95 * len(sorted_latency)) - 1] if sorted_latency else None,
            "over_300_sec": sum(t > 300 for t in latencies)}
