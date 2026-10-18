import copy
import unittest

from app.rules import RuleInputError, finalize_detection, load_registry
from test_rules import fixture


class DetectionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.report, self.packet, self.registry = fixture()
        self.report['job_id'] = 'synthetic-job'
        self.scope = [self.packet['rule_id']]
        self.search = dict(schema_version='detector_execution.v1', job_id='synthetic-job',
                           detector_runs=[dict(rule_id=self.scope[0], status='complete',
                                               detector_version='synthetic-only',
                                               processed_window_ids=[0], refinement_status='completed')])

    def run_policy(self, packets=None):
        return finalize_detection(self.report, [self.packet] if packets is None else packets,
                                  self.search, supported_rule_ids=self.scope, registry=self.registry)

    def test_positive(self):
        result = self.run_policy()
        self.assertEqual(result['verdict'], 'violation')
        self.assertEqual(len(result['supported']), 1)
        self.assertFalse(result['public_verdict_ready'])

    def test_empty_requires_executed_detector(self):
        self.assertEqual(self.run_policy([])['verdict'], 'no_violation')
        self.search['detector_runs'] = []
        with self.assertRaises(RuleInputError):
            self.run_policy([])

    def test_unknown_remains_unknown_and_does_not_support(self):
        self.packet['facts']['service_completed'] = None
        original = copy.deepcopy(self.packet)
        result = self.run_policy()
        self.assertEqual(result['verdict'], 'no_violation')
        self.assertEqual(len(result['unresolved']), 1)
        self.assertEqual(self.packet, original)

    def test_unknown_does_not_cancel_other_positive(self):
        unknown = copy.deepcopy(self.packet)
        unknown['event_id'] = 'unknown-episode'
        unknown['facts']['service_completed'] = None
        result = self.run_policy([self.packet, unknown])
        self.assertEqual(result['verdict'], 'violation')
        self.assertEqual(len(result['unresolved']), 1)

    def test_conflict_is_retained(self):
        self.packet['facts']['service_completed'] = True
        result = self.run_policy()
        self.assertEqual(result['supported'], [])
        self.assertEqual(result['unresolved'][0]['reasons'], ['conflicting_service_state'])

    def test_hard_negative(self):
        self.packet['facts']['cashier_left_workplace'] = False
        self.assertEqual(self.run_policy()['verdict'], 'no_violation')

    def test_incomplete_or_unexecuted_run_never_negative(self):
        for field, value in [('status', 'failed'), ('processed_window_ids', []),
                             ('processed_window_ids', [0, 0]), ('processed_window_ids', [99]),
                             ('refinement_status', 'pending'), ('detector_version', '')]:
            with self.subTest(field=field, value=value):
                previous = copy.deepcopy(self.search)
                self.search['detector_runs'][0][field] = value
                with self.assertRaises(RuleInputError):
                    self.run_policy([])
                self.search = previous

    def test_wrong_job_or_partial_report(self):
        self.search['job_id'] = 'other-job'
        with self.assertRaises(RuleInputError):
            self.run_policy()
        self.search['job_id'] = self.report['job_id']
        self.report['coverage_scope'] = 'interval'
        with self.assertRaises(RuleInputError):
            self.run_policy()
        self.report['coverage_scope'] = 'full_video'
        self.report['complete'] = False
        with self.assertRaises(RuleInputError):
            self.run_policy()

    def test_unreviewed_scope_never_negative(self):
        self.registry = load_registry()
        with self.assertRaises(RuleInputError):
            self.run_policy([])

    def test_out_of_scope_packet(self):
        self.packet['rule_id'] = 'personal_items_storage'
        with self.assertRaises(RuleInputError):
            self.run_policy()

    def test_missing_or_duplicate_scope(self):
        for scope in ([], self.scope * 2):
            self.scope = scope
            with self.assertRaises(RuleInputError):
                self.run_policy([])
