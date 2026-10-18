"""Validate A's bundle before B's scoped policy. Not a production factory.

The result remains internal until release validation and source JPEG checks.
Never log this function's inputs or outputs.
"""
from app.rules import finalize_detection


def _require(condition):
    if not condition:
        raise ValueError("Invalid assessment integration contract")


def finalize_bundle(original_report, bundle, *, supported_rule_ids, registry=None):
    """Reject unexecuted prototypes and mismatched reports before finalization.

An execution receipt is supplied by A, never synthesized by this adapter.
Its authenticity/quality still requires validation of A's producer locally.
"""
    _require(isinstance(bundle, dict))
    _require(bundle.get("schema_version") == "assessment_bundle.v1")
    _require(bundle.get("complete") is True)
    report = bundle.get("report")
    _require(isinstance(report, dict))
    for field in ("job_id", "video", "model"):
        _require(field in original_report and report.get(field) == original_report[field])
    windows = report.get("windows")
    _require(isinstance(windows, list) and all(isinstance(w, dict) for w in windows))
    ids = [w.get("window_id") for w in windows]
    _require(all(type(wid) is int for wid in ids) and len(set(ids)) == len(ids))
    by_id = {w["window_id"]: w for w in windows}
    for window in original_report["windows"]:
        _require(by_id.get(window["window_id"]) == window)
    diagnostics = bundle.get("diagnostics")
    _require(isinstance(diagnostics, dict))
    search = diagnostics.get("search")
    # A supplies this receipt; B additionally checks runs and window coverage.
    _require(isinstance(search, dict) and search.get("schema_version") == "detector_execution.v1")
    return finalize_detection(report, bundle.get("assessments"), search,
                              supported_rule_ids=supported_rule_ids, registry=registry)
