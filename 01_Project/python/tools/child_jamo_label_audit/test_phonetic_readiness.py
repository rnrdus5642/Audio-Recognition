"""Scientific failure cases: notes, missing contrasts, crops and leakage."""
import copy
import unittest
from unittest.mock import patch

from analyze_phonetic import assess
from inventory_phonetic import DEST
from common import load
import labels


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.evidence = load(DEST / 'evidence.json')
        self.snapshot = {'latest_revision': 15, 'items': []}
        for e in self.evidence:
            self.snapshot['items'].append({'id': str(e['number']), 'binding': e['binding'],
                'reveal': {'first_heard': 'an approximate note, not a label'},
                'answer': {'revision': e['number'], 'choice': 'ambiguous', 'note': '', 'crop_relation': ''}})
        self.anchor = next(e for e in self.evidence if e['case'] == 6 and e['role'] == 'relabel_train')
        self.dev = [e for e in self.evidence if e['case'] == 6 and e['role'] == 'matched_development']

    def choose(self, entry, original):
        key = next(k for k, c in entry['candidates'].items() if c['is_original_training_target'] == original)
        reply = next(r for r in self.snapshot['items'] if r['id'] == str(entry['number']))
        reply['answer']['choice'] = key
        return reply

    def test_notes_and_partial_answers_never_become_targets(self):
        for entry, reply in zip(self.evidence, self.snapshot['items']):
            reply['answer']['note'] = next(iter(entry['candidates'].values()))['surface']
            reply['answer']['choice'] = 'partial'
        report = assess(self.evidence, self.snapshot)
        self.assertFalse(report['ready_for_controlled_comparison'])
        self.assertTrue(all(r['confirmed_ids'] is None for r in report['items']))

    def test_two_same_prompt_original_forms_are_not_variant_coverage(self):
        self.choose(self.anchor, False)
        for entry in self.dev:
            self.choose(entry, True)
        report = assess(self.evidence, self.snapshot)
        self.assertFalse(report['ready_for_controlled_comparison'])
        self.assertEqual(report['next_action'], 'obtain_missing_contrast_coverage_before_training')

    def test_independent_original_and_same_variant_enable_only_that_case(self):
        self.choose(self.anchor, False)
        self.choose(self.dev[0], True)
        self.choose(self.dev[1], False)
        with patch.object(labels, 'targets', side_effect=AssertionError('Never G2P a confirmed sound form')):
            report = assess(self.evidence, self.snapshot)
        self.assertTrue(report['ready_for_controlled_comparison'])
        self.assertEqual(report['qualified_comparison_cases'], [6])
        self.assertEqual(report['confirmed_training_relabel_questions'], [self.anchor['number']])

    def test_crop_uncertainty_excludes_an_otherwise_confirmed_form(self):
        self.choose(self.anchor, False)['answer']['crop_relation'] = 'uncertain'
        self.choose(self.dev[0], True)
        self.choose(self.dev[1], False)
        report = assess(self.evidence, self.snapshot)
        self.assertFalse(report['ready_for_controlled_comparison'])
        anchor = next(r for r in report['items'] if r['number'] == self.anchor['number'])
        self.assertIsNone(anchor['confirmed_ids'])

    def test_pending_questions_block_comparison_even_with_one_covered_case(self):
        self.choose(self.anchor, False)
        self.choose(self.dev[0], True)
        self.choose(self.dev[1], False)
        self.snapshot['items'][0]['answer'] = None
        self.assertFalse(assess(self.evidence, self.snapshot)['ready_for_controlled_comparison'])

    def test_wrong_binding_or_speaker_leakage_is_rejected(self):
        bad = copy.deepcopy(self.snapshot)
        bad['items'][0]['binding'] = 'different recording'
        with self.assertRaises(AssertionError):
            assess(self.evidence, bad)
        self.dev[0]['row']['speaker'] = self.anchor['row']['speaker']
        with self.assertRaises(AssertionError):
            assess(self.evidence, self.snapshot)


if __name__ == '__main__':
    unittest.main()
