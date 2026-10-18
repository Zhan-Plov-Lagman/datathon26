import copy
from pathlib import Path
import tempfile
import unittest

from evaluation.note_review import group_notes, suggest_cases, reviewed_records


class NoteReviewTests(unittest.TestCase):
    def test_hints_are_not_labels_and_duplicates_not_selected_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'X-1.mov';b=Path(tmp)/'Х-2.MOV';c=Path(tmp)/'X-3.mov'
            a.write_bytes(b'synthetic-1');b.write_bytes(b'synthetic-2');c.write_bytes(a.read_bytes())
            grouped=group_notes('ФАЙЛ: X-1\nСинтетика: уходит\n**Х-2**\nСинтетика: встаёт\nX-3\nкопия',[a,b,c])
            cases,unique=suggest_cases(grouped)
            self.assertEqual(unique,2);self.assertEqual(len(cases),2)
            self.assertTrue(all(c['label'] is None and c['reviewed'] is False for c in cases))
            with self.assertRaises(ValueError):reviewed_records({'cases':cases})
            for c in cases:c.update(reviewed=True,reviewer='synthetic',label=False)
            self.assertEqual(len(reviewed_records({'cases':cases})),2)
            cases[0]['label']=None
            with self.assertRaises(ValueError):reviewed_records({'cases':cases})

    def test_review_requires_consistent_finite_event_and_unchanged_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'X-1.mov';p.write_bytes(b'synthetic')
            cases,_=suggest_cases([(p,'synthetic')])
            c=cases[0];c.update(reviewed=True,reviewer='synthetic',label=True)
            with self.assertRaises(ValueError):reviewed_records({'cases':cases})
            c['events']=[{'type':'cashier_leaves_service','timecode_sec':1.5}]
            self.assertTrue(reviewed_records({'cases':cases})[0]['label'])
            bad=copy.deepcopy(cases);bad[0]['events'][0]['timecode_sec']=float('nan')
            with self.assertRaises(ValueError):reviewed_records({'cases':bad})
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):reviewed_records({'cases':cases})


if __name__=='__main__':unittest.main()
