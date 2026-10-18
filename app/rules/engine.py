"""Consume observations.v1 and separately grounded fact assessments.

Free-text observations never become predicates by keyword matching. All content
returned by this module is local-only. Callers must not log scene descriptions.
"""
import hashlib
import json
import math
from pathlib import Path


class RuleInputError(ValueError):
    pass


def require(condition):
    if not condition:
        raise RuleInputError("Invalid rule input; inspect locally")


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def load_registry(path=None):
    # JSON is a YAML 1.2 subset; avoid introducing shared runtime dependencies.
    path = Path(path) if path else Path(__file__).resolve().parents[2] / "config" / "rules.yaml"
    registry = json.loads(path.read_text(encoding="utf-8"))
    validate_registry(registry)
    return registry


def validate_registry(registry):
    require(registry.get("schema_version") == "rules.v1")
    require(isinstance(registry.get("version"), str))
    require(isinstance(registry.get("rules"), list))
    ids = set()
    for rule in registry["rules"]:
        require(isinstance(rule.get("rule_id"), str) and rule["rule_id"] not in ids)
        ids.add(rule["rule_id"])
        require(rule.get("status") in ("hypothesis", "reviewed", "retired"))
        for key in ("type", "title", "explanation"):
            require(isinstance(rule.get(key), str) and bool(rule[key].strip()))
        for key in ("applicability", "required_facts", "exceptions"):
            require(isinstance(rule.get(key), list))
            require(all(isinstance(x, str) and x for x in rule[key]))
        require(bool(rule["applicability"]) and bool(rule["required_facts"]))
        if rule["status"] == "reviewed":
            source = rule.get("source", {})
            for key in ("document", "clause", "url", "edition_date", "checked_on", "reviewer"):
                require(isinstance(source.get(key), str) and bool(source[key].strip()))
            require(source["url"].startswith("https://"))
            require(rule.get("validated_on_local_cases") is True)


def validate_report(report):
    require(report.get("schema_version") == "observations.v1")
    require(report.get("analysis_kind") == "observations_only")
    require(report.get("complete") is True)
    require(report.get("coverage_scope") in ("full_video", "interval"))
    duration = report.get("video", {}).get("duration_sec")
    require(number(duration) and duration > 0)
    require(isinstance(report.get("windows"), list) and bool(report["windows"]))
    windows = {}
    for window in report["windows"]:
        wid = window.get("window_id")
        require(type(wid) is int and wid >= 0 and wid not in windows)
        start, end = window.get("window_start_sec"), window.get("window_end_sec")
        require(number(start) and number(end) and 0 <= start <= end <= duration)
        times = window.get("frame_timestamps")
        require(isinstance(times, list) and bool(times))
        require(all(number(t) and 0 <= t <= duration for t in times))
        require(all(a < b for a, b in zip(times, times[1:])))
        require(isinstance(window.get("visibility_limits"), list))
        require(isinstance(window.get("observations"), list))
        for obs in window["observations"]:
            a, b = obs.get("start_frame"), obs.get("end_frame")
            require(type(a) is int and type(b) is int and 0 <= a <= b < len(times))
            require(obs.get("start_sec") == times[a] and obs.get("end_sec") == times[b])
        windows[wid] = window
    return windows


def evaluate(report, assessments, registry=None):
    """Evaluate proposed local fact packets (see docs/RULES_B.md).

    This intentionally does not return a public no_violation verdict. Missing
    fact extraction / unknown exceptions / unreviewed law are unresolved work.
    """
    return _evaluate(report,assessments,registry)


def validate_signs_registry(registry, supported_rule_ids):
    """Verify public source text and explicit scope, without claiming detector validation."""
    validate_registry(registry)
    require(isinstance(supported_rule_ids,(list,tuple)) and bool(supported_rule_ids))
    require(all(isinstance(rule_id,str) for rule_id in supported_rule_ids))
    require(len(set(supported_rule_ids))==len(supported_rule_ids))
    rules={r['rule_id']:r for r in registry['rules']}
    for rule_id in supported_rule_ids:
        require(rule_id in rules)
        rule=rules[rule_id]
        require(rule['status'] in ('hypothesis','reviewed'))
        source=rule.get('source',{})
        require(source.get('text_verified') is True)
        for key in ('document','clause','url'):
            require(isinstance(source.get(key),str) and bool(source[key].strip()))
        require(source['url'].startswith('https://') and len(source['url'])>8)


def evaluate_signs(report, assessments, registry, *, supported_rule_ids):
    """Evaluate possible signs under verified public sources; accuracy remains unmeasured."""
    validate_signs_registry(registry,supported_rule_ids)
    return _evaluate(report,assessments,registry,signs_rule_ids=set(supported_rule_ids))


def _evaluate(report, assessments, registry=None, *, signs_rule_ids=None):
    registry = load_registry() if registry is None else registry
    validate_registry(registry)
    windows = validate_report(report)
    require(isinstance(assessments, list))
    rules = {r["rule_id"]: r for r in registry["rules"]}
    results, seen = [], set()
    for packet in assessments:
        require(isinstance(packet, dict))
        require(packet.get("rule_id") in rules)
        rule = rules[packet["rule_id"]]
        if signs_rule_ids is not None:
            require(rule['rule_id'] in signs_rule_ids)
        event_id = packet.get("event_id")
        require(isinstance(event_id, str) and bool(event_id))
        identity = (rule["rule_id"], event_id)
        # One packet per event after refinement. Never merge two episodes just
        # because they overlap in time or share a class.
        require(identity not in seen)
        seen.add(identity)
        refs = packet.get("observation_refs")
        require(isinstance(refs, list) and bool(refs))
        observations, timestamps, limits = [], set(), False
        for ref in refs:
            require(isinstance(ref, dict))
            wid, index = ref.get("window_id"), ref.get("observation_index")
            require(type(wid) is int and wid in windows and type(index) is int)
            window = windows[wid]
            require(0 <= index < len(window["observations"]))
            obs = window["observations"][index]
            observations.append(obs)
            timestamps.update(window["frame_timestamps"][obs["start_frame"]:obs["end_frame"] + 1])
            limits |= bool(window["visibility_limits"])
        start = min(o["start_sec"] for o in observations)
        end = max(o["end_sec"] for o in observations)
        anchor = packet.get("anchor_sec")
        require(number(anchor) and anchor in timestamps)
        facts = packet.get("facts")
        require(isinstance(facts, dict))
        known = set(rule["applicability"] + rule["required_facts"] + rule["exceptions"])
        require(set(facts).issubset(known))
        require(all(v is None or type(v) is bool for v in facts.values()))
        positive = rule["applicability"] + rule["required_facts"]
        contradicted = [k for k in positive if facts.get(k) is False]
        exceptions = [k for k in rule["exceptions"] if facts.get(k) is True]
        missing = [k for k in positive + rule["exceptions"] if facts.get(k) is None]
        reasons = []
        if signs_rule_ids is None and rule["status"] != "reviewed":
            reasons.append("rule_not_reviewed")
        if missing:
            reasons.append("unknown_facts_or_exceptions")
        if packet.get("refinement_complete") is not True:
            reasons.append("refinement_required")
        if limits:
            reasons.append("visibility_limited")
        if rule.get("requires_full_sequence") and packet.get("full_sequence_visible") is not True:
            reasons.append("full_sequence_required")
        decision = "rejected" if contradicted or exceptions else "candidate" if reasons else "supported"
        result = {"rule_id": rule["rule_id"], "event_id": event_id, "type": rule["type"],
                  "start_sec": start, "end_sec": end, "anchor_sec": anchor,
                  "observation_refs": refs, "checked_conditions": dict(facts),
                  "exceptions": exceptions, "contradicted": contradicted,
                  "missing": missing, "decision": decision, "reasons": reasons}
        if decision == "supported":
            source = rule["source"]
            explanation=(rule['explanation'] if signs_rule_ids is None else
                f'Обнаружены признаки возможного нарушения: {rule["title"]}.')
            result["description"] = (f'{explanation} Основание: {source["document"]}, '
                                     f'п. {source["clause"]}. {source["url"]}')
        results.append(result)
    return {"schema_version": "rule_results.v1", "job_id": report.get("job_id"),
            "registry_version": registry["version"],
            "registry_sha256": hashlib.sha256(json.dumps(registry, sort_keys=True).encode()).hexdigest(),
            "model": report.get("model"), "coverage_scope": report["coverage_scope"],
            "candidates": results, "supported": [r for r in results if r["decision"] == "supported"],
            "requires_review": not assessments or any(r["decision"] == "candidate" for r in results),
            "public_verdict_ready": False}
