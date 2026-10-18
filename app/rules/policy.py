"""Detection policy; internal result, not C's production RuleDecision."""
from .engine import evaluate, load_registry, require, validate_report


POLICY_VERSION = "detection-policy.v1"


def finalize_detection(report, assessments, search, *, supported_rule_ids, registry=None):
    """Require a trusted A execution receipt, then finalize scoped detections.

    Receipts must be produced by executed detector work, never inferred from
    empty assessments. This validates their structure, not their truthfulness.
    C still owns release readiness, provenance checks and JPEG validation.
    """
    registry = load_registry() if registry is None else registry
    windows = validate_report(report)
    require(report["coverage_scope"] == "full_video")
    require(isinstance(report.get("job_id"), str) and bool(report["job_id"]))
    require(isinstance(supported_rule_ids, list) and bool(supported_rule_ids))
    require(all(isinstance(x, str) and x for x in supported_rule_ids))
    require(len(set(supported_rule_ids)) == len(supported_rule_ids))
    rules = {r["rule_id"]: r for r in registry["rules"]}
    require(set(supported_rule_ids).issubset(rules))
    # A disabled rule cannot become an apparently successful negative detector.
    require(all(rules[r]["status"] == "reviewed" and
                rules[r].get("validated_on_local_cases") is True
                for r in supported_rule_ids))
    require(isinstance(search, dict))
    require(search.get("schema_version") == "detector_execution.v1")
    require(search.get("job_id") == report["job_id"])
    runs = search.get("detector_runs")
    require(isinstance(runs, list) and len(runs) == len(supported_rule_ids))
    seen = set()
    for run in runs:
        require(isinstance(run, dict))
        rule_id = run.get("rule_id")
        require(isinstance(rule_id, str) and rule_id in supported_rule_ids and rule_id not in seen)
        seen.add(rule_id)
        require(run.get("status") == "complete")
        require(isinstance(run.get("detector_version"), str) and bool(run["detector_version"].strip()))
        ids = run.get("processed_window_ids")
        require(isinstance(ids, list) and all(type(x) is int for x in ids))
        require(len(ids) == len(set(ids)) and set(ids) == set(windows))
        require(run.get("refinement_status") in ("completed", "not_needed"))
    require(isinstance(assessments, list))
    require(all(isinstance(p, dict) and p.get("rule_id") in supported_rule_ids for p in assessments))
    result = evaluate(report, assessments, registry)
    unresolved = []
    supported = []
    for event in result["candidates"]:
        facts = event["checked_conditions"]
        conflict = (event["rule_id"] == "cashier_leaves_service" and
                    facts.get("service_in_progress") is True and
                    facts.get("service_completed") is True)
        if conflict or event["reasons"]:
            unresolved.append({"rule_id": event["rule_id"], "event_id": event["event_id"],
                               "reasons": ["conflicting_service_state"] if conflict else event["reasons"]})
        elif event["decision"] == "supported":
            supported.append(event)
    return {"schema_version": "detection_result.v1", "job_id": report["job_id"],
            "complete": True, "policy_version": POLICY_VERSION,
            "negative_policy_version": POLICY_VERSION,
            "supported_rule_ids": list(supported_rule_ids),
            "registry_version": result["registry_version"],
            "registry_sha256": result["registry_sha256"],
            "verdict": "violation" if supported else "no_violation",
            "supported": supported, "unresolved": unresolved,
            "public_verdict_ready": False}
