import copy

import pytest

from app.inference.assessments import build_bundle
from app.jobs.detection import finalize_bundle
from app.rules import RuleInputError
from tests.rules.test_rules import fixture


def setup_bundle():
    report, packet, registry = fixture()
    report['job_id'] = 'synthetic-job'
    report.setdefault('model', {'id': 'synthetic', 'revision': 'test'})
    bundle = dict(schema_version='assessment_bundle.v1', complete=True,
                  report=copy.deepcopy(report), assessments=[packet], diagnostics={'search': {
                      'schema_version': 'detector_execution.v1', 'job_id': report['job_id'],
                      'detector_runs': [dict(rule_id=packet['rule_id'], status='complete',
                                            detector_version='synthetic', processed_window_ids=[0],
                                            refinement_status='completed')]}, 'episodes': [{
                      'rule_id': packet['rule_id'], 'event_id': packet['event_id'], 'conflicts': [],
                      'fact_sources': {name: {'origin': 'visual',
                                             'observation_refs': copy.deepcopy(packet['observation_refs'])}
                                       for name in packet['facts']}}]})
    return report, bundle, registry, [packet['rule_id']]


@pytest.mark.parametrize('unknown', [False, True])
def test_two_verdicts_remain_internal(unknown):
    report, bundle, registry, scope = setup_bundle()
    if unknown:
        bundle['assessments'][0]['facts']['service_completed'] = None
        bundle['diagnostics']['episodes'][0]['fact_sources']['service_completed'] = {
            'origin': 'unknown', 'observation_refs': [], 'reason': 'synthetic uncertainty'}
    before = copy.deepcopy(bundle)
    result = finalize_bundle(report, bundle, registry=registry, supported_rule_ids=scope)
    assert result['verdict'] == ('no_violation' if unknown else 'violation')
    assert result['public_verdict_ready'] is False
    assert bundle == before


def test_unexecuted_a_bundle_cannot_be_negative():
    report, _, registry, scope = setup_bundle()
    # Real A bundle builder, synthetic observations only.
    results = [dict(operation_context='synthetic', visibility_limits=[], observations=[], episodes=[])]
    prototype = build_bundle(report, results)
    # A now supplies the schema even without execution, but detector_runs is
    # empty. B must reject that receipt rather than infer a negative verdict.
    with pytest.raises(RuleInputError):
        finalize_bundle(report, prototype, registry=registry, supported_rule_ids=scope)


@pytest.mark.parametrize('change', ['job', 'window', 'receipt', 'disabled'])
def test_broken_contract_is_never_negative(change):
    report, bundle, registry, scope = setup_bundle()
    bundle['assessments'] = []
    if change == 'job':
        bundle['report']['job_id'] = 'other'
    elif change == 'window':
        bundle['report']['windows'][0]['visibility_limits'] = ['changed']
    elif change == 'receipt':
        bundle['diagnostics']['search']['detector_runs'][0]['processed_window_ids'] = []
    else:
        for rule in registry['rules']:
            rule['status'] = 'hypothesis'
    with pytest.raises((ValueError, RuleInputError)):
        finalize_bundle(report, bundle, registry=registry, supported_rule_ids=scope)
