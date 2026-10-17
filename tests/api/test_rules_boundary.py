"""Actual B engine at C's boundary; only synthetic observations and policies."""
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.api.schema import AnalysisResult
from app.jobs.pipeline import IntegratedProcessor, RuleDecision
from app.rules import evaluate
from tests.rules.test_rules import fixture


@pytest.mark.parametrize("with_assessment", [False, True])
def test_current_b_result_cannot_become_public_verdict(with_assessment):
    report, packet, _ = fixture()
    decision = evaluate(report, [packet] if with_assessment else [])
    assert decision["public_verdict_ready"] is False
    assert decision["supported"] == []
    with pytest.raises(ValidationError):
        RuleDecision.model_validate(decision)
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(decision)


@pytest.mark.parametrize("policy", [None, "", "   "])
def test_private_result_rejects_unjustified_negative(result, policy):
    result.update(verdict="no_violation", violations=[], rule_ids=[],
                  negative_policy_version=policy)
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(result)


@pytest.mark.parametrize("policy", [None, "synthetic-test-only"])
def test_empty_adapter_requires_policy_before_evidence(policy):
    report, _, _ = fixture()
    report["model"] = {"id": "synthetic", "revision": "test"}
    decision = dict(complete=True, rules_reviewed=True, rules_version="synthetic",
                    supported=[], negative_policy_version=policy)
    # No evidence method: negative decisions must not request frames.
    worker = SimpleNamespace(analyze=lambda *args, **kwargs: report)
    rules = SimpleNamespace(reviewed=True, evaluate=lambda *args, **kwargs: decision)
    processor = IntegratedProcessor(worker, rules)
    if policy is None:
        with pytest.raises(ValidationError):
            processor.process("synthetic.mov", "test", 10)
    else:
        result = AnalysisResult.model_validate(processor.process("synthetic.mov", "test", 10))
        assert result.negative_policy_version == policy
        assert result.public("test") == {
            "job_id": "test", "status": "done", "verdict": "no_violation", "violations": []}


def test_positive_result_does_not_need_negative_policy(result):
    assert AnalysisResult.model_validate(result).verdict == "violation"
