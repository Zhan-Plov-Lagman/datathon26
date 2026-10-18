import unittest
from app.perception.detectors import Detection
from app.perception.temporal import Tracker,plan_reviews


class TemporalTests(unittest.TestCase):
    def test_one_to_one_ids_and_reset_after_gap(self):
        p=lambda box:Detection('person',.9,box)
        tracker=Tracker(max_gap_sec=.6)
        first=tracker.update(0,[p((.1,.1,.4,.9)),p((.5,.1,.8,.9))])
        ids={t.id for t in first}
        again=tracker.update(.2,[p((.12,.1,.42,.9)),p((.48,.1,.78,.9))])
        self.assertEqual({t.id for t in again},ids)
        self.assertEqual(len(again),len(ids))
        tracker.update(.5,[])
        returned=tracker.update(1.2,[p((.12,.1,.42,.9))])
        self.assertNotIn(returned[0].id,ids)
        with self.assertRaises(ValueError):
            tracker.update(1.2,[])

    def test_full_intervals_are_preserved_and_periodic_static_check_exists(self):
        signals=[(.1,'movement',1),(9.,'movement',1),(18.,'object_state',None),(39.9,'movement',2)]
        windows=plan_reviews(40,signals)
        self.assertTrue(any('periodic_context' in w.reasons for w in windows))
        self.assertTrue(all(0<=w.start_sec<w.end_sec<=40 and w.end_sec-w.start_sec<=10 for w in windows))
        for t,reason,actor in signals:
            self.assertTrue(any(w.start_sec<=t<=w.end_sec and reason in w.reasons
                                and (actor is None or actor in w.actor_ids) for w in windows))
        self.assertTrue(any(w.start_sec==0 for w in windows))
        self.assertTrue(any(w.end_sec==40 for w in windows))
