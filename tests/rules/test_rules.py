import copy
import unittest

from app.rules import RuleInputError, evaluate, load_registry


def fixture():
    registry = load_registry()
    rule = registry['rules'][0]
    rule['status'] = 'reviewed'
    rule['validated_on_local_cases'] = True
    rule['source'].update(edition_date='2026-01-01', checked_on='2026-10-17', reviewer='SYNTHETIC TEST ONLY')
    report = dict(schema_version='observations.v1', analysis_kind='observations_only', complete=True,
                  coverage_scope='full_video', video={'duration_sec': 10}, windows=[
        dict(window_id=0, window_start_sec=0, window_end_sec=10, frame_timestamps=[0, 2, 4, 6],
             visibility_limits=[], observations=[dict(start_frame=1, end_frame=2, start_sec=2,
                                                       end_sec=4, description='Synthetic action')])])
    packet = dict(rule_id=rule['rule_id'], event_id='synthetic-episode', anchor_sec=2,
                  observation_refs=[dict(window_id=0, observation_index=0)],
                  facts={k: True for k in rule['applicability'] + rule['required_facts']},
                  refinement_complete=True)
    packet['facts'].update({k: False for k in rule['exceptions']})
    return report, packet, registry


class RuleTests(unittest.TestCase):
    def test_supported_and_registry_citation(self):
        report, packet, registry = fixture()
        result = evaluate(report, [packet], registry)
        self.assertEqual(len(result['supported']), 1)
        self.assertIn(registry['rules'][0]['source']['url'], result['supported'][0]['description'])
        self.assertFalse(result['public_verdict_ready'])

    def test_production_registry_cannot_support(self):
        report, packet, _ = fixture()
        self.assertEqual(evaluate(report, [packet])['supported'], [])

    def test_unknown_exception_is_not_false(self):
        report, packet, registry = fixture()
        del packet['facts']['service_completed']
        self.assertEqual(evaluate(report, [packet], registry)['candidates'][0]['decision'], 'candidate')

    def test_exception_and_contradiction(self):
        for fact in ('service_completed', 'cashier_left_workplace'):
            report, packet, registry = fixture()
            packet['facts'][fact] = fact == 'service_completed'
            self.assertEqual(evaluate(report, [packet], registry)['candidates'][0]['decision'], 'rejected')

    def test_occlusion_requires_review(self):
        report, packet, registry = fixture()
        report['windows'][0]['visibility_limits'] = ['Synthetic occlusion']
        self.assertEqual(evaluate(report, [packet], registry)['supported'], [])

    def test_omission_requires_full_sequence(self):
        report, packet, registry = fixture()
        registry['rules'][0]['requires_full_sequence'] = True
        self.assertEqual(evaluate(report, [packet], registry)['supported'], [])
        packet['full_sequence_visible'] = True
        self.assertEqual(len(evaluate(report, [packet], registry)['supported']), 1)

    def test_failure_never_negative(self):
        report, packet, registry = fixture()
        report['complete'] = False
        with self.assertRaises(RuleInputError):
            evaluate(report, [packet], registry)

    def test_bad_anchor_reference_and_fact(self):
        for field, value in [('anchor_sec', float('nan')), ('anchor_sec', 3),
                             ('facts', {'service_completed': 'false'}),
                             ('observation_refs', [{'window_id': 0, 'observation_index': -1}])]:
            report, packet, registry = fixture()
            packet[field] = value
            with self.assertRaises(RuleInputError):
                evaluate(report, [packet], registry)

    def test_empty_text_not_negative(self):
        report, _, registry = fixture()
        result = evaluate(report, [], registry)
        self.assertTrue(result['requires_review'])
        self.assertNotIn('verdict', result)

    def test_distinct_episodes_and_duplicate_packet(self):
        report, packet, registry = fixture()
        other = copy.deepcopy(packet)
        other['event_id'] = 'another-episode'
        self.assertEqual(len(evaluate(report, [packet, other], registry)['supported']), 2)
        with self.assertRaises(RuleInputError):
            evaluate(report, [packet, packet], registry)

    def test_time_must_match_source_frame(self):
        report, packet, registry = fixture()
        report['windows'][0]['observations'][0]['start_sec'] = 1.9
        with self.assertRaises(RuleInputError):
            evaluate(report, [packet], registry)

    def test_reviewed_rule_requires_source_and_validation(self):
        report, packet, registry = fixture()
        registry['rules'][0]['source']['edition_date'] = None
        with self.assertRaises(RuleInputError):
            evaluate(report, [packet], registry)


if __name__ == '__main__':
    unittest.main()
