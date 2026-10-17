import copy
import unittest

from evaluation.dataset import make_split, prepare, select_records
from evaluation.metrics import match_events, score


def event(t, kind='synthetic'):
    return {'type': kind, 'timecode_sec': t}


class EvaluationTests(unittest.TestCase):
    def test_one_to_one_and_maximum_matching(self):
        self.assertEqual(len(match_events([event(1)], [event(1), event(1)])), 1)
        self.assertEqual(len(match_events([event(1), event(0)], [event(0), event(2)], 1)), 2)

    def test_classification_is_separate(self):
        self.assertEqual(match_events([event(1)], [event(1, 'other')]), [])
        self.assertEqual(len(match_events([event(1)], [event(1, 'other')], typed=False)), 1)

    def test_failed_missing_and_ambiguous_visible(self):
        rows = [dict(id=str(i), sha256=str(i), label=True, reviewed=True,
                     events=[event(1)], ambiguous=i == 3) for i in range(4)]
        predictions = [dict(id='0', status='done', verdict='violation', violations=[event(1), event(1)], latency_sec=4),
                       dict(id='1', status='failed', latency_sec=301)]
        result = score(rows, predictions)
        self.assertEqual(result['completion_rate'], 1/3)
        self.assertEqual(result['events']['tp'], 1)
        self.assertEqual(result['events']['fp'], 1)
        self.assertEqual(result['events']['fn'], 2)
        self.assertEqual(result['over_300_sec'], 1)
        self.assertEqual(result['videos']['ambiguous'], 1)
        self.assertEqual(result['per_class']['synthetic']['fn'], 2)

    def test_invalid_and_empty_metrics(self):
        self.assertIsNone(score([], [])['videos']['f1'])
        with self.assertRaises(ValueError):
            match_events([event(float('nan'))], [])
        with self.assertRaises(ValueError):
            score([], [{'id': 'unknown'}])

    def test_group_transitivity_and_determinism(self):
        rows = [dict(file=str(i), sha256=str(i), scene_group=str(i), label=bool(i % 2)) for i in range(12)]
        rows[1].update(sha256='0', label=False)
        rows[2]['scene_group'] = '1'
        manifest = dict(scene_grouping_reviewed=True, records=rows)
        result = make_split(manifest)
        locations = {f: split for split, groups in result['groups'].items() for group in groups for f in group}
        self.assertEqual(locations['0'], locations['2'])
        reversed_manifest = copy.deepcopy(manifest)
        reversed_manifest['records'].reverse()
        self.assertEqual(result['groups'], make_split(reversed_manifest)['groups'])
        self.assertTrue(all(result['groups'].values()))

    def test_unreviewed_grouping_blocked(self):
        with self.assertRaises(ValueError):
            make_split(dict(records=[], scene_grouping_reviewed=False))

    def test_preserve_conflicting_original_timestamps(self):
        videos = [dict(file=str(i), sha256='same', duration_sec=10, label='original',
                       timestamp_raw=str(i), timestamp_sec=i) for i in (1, 2)]
        result = prepare(dict(issues=[], videos=videos))
        self.assertTrue(all(r['ambiguous'] for r in result['records']))
        self.assertEqual([r['organizer_timestamp_sec'] for r in result['records']], [1, 2])

    def test_frozen_manifest_and_single_representative(self):
        rows = [dict(file=str(i), sha256=str(i), scene_group=str(i), label=False,
                     reviewed=True, ambiguous=False, events=[]) for i in range(10)]
        rows[1]['sha256'] = '0'
        manifest = dict(scene_grouping_reviewed=True, records=rows)
        split = make_split(manifest)
        selected = [r for part in split['groups'] for r in select_records(manifest, split, part)['records']]
        self.assertEqual(len(selected), 9)
        rows[0]['events'] = [event(1)]
        with self.assertRaises(ValueError):
            select_records(manifest, split, 'development')


if __name__ == '__main__':
    unittest.main()
