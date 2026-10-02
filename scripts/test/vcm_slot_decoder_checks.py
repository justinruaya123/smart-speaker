"""Behavioral checks for the isolated E19 semantic output boundary."""
import unittest
from vcm_slot_decoder import catalog_fingerprint, number_entries, time_entries, decision, calibrate


class SlotDecoderChecks(unittest.TestCase):
    def setUp(self):
        self.entries = number_entries()
        self.version = catalog_fingerprint(self.entries)
        self.gate = dict(passed=True, threshold=.7, margin=.1, catalog_version=self.version)

    def test_numeric_composition_inventory(self):
        by_value = {x['value']: x for x in self.entries}
        self.assertEqual(set(by_value), set(range(101)))
        self.assertIn('seventy five', by_value[75]['aliases'])
        self.assertIn('one hundred percent', by_value[100]['aliases'])
        self.assertIn('thirty seven', by_value[37]['aliases'])

    def test_no_intent_rescue(self):
        result = decision(self.entries, [('number:75', .99), ('number:25', .1)],
                          'UNKNOWN', 'number', self.gate)
        self.assertEqual(result['reason'], 'intent_does_not_own_slot')

    def test_stale_catalog_invalidates_acceptance(self):
        self.gate['catalog_version'] = 'stale'
        result = decision(self.entries, [('number:75', .99), ('number:25', .1)],
                          'BRIGHTNESS', 'number', self.gate)
        self.assertEqual(result['reason'], 'catalog_requires_calibration')

    def test_close_or_tied_candidates_reject(self):
        for scores in ((.9,.9), (.9,.85)):
            result = decision(self.entries, list(zip(['number:75','number:25'], scores)),
                              'BRIGHTNESS', 'number', self.gate)
            self.assertNotEqual(result['status'], 'accepted')

    def test_missing_meridiem_never_invents_time(self):
        entries = time_entries()
        gate = dict(self.gate, catalog_version=catalog_fingerprint(entries))
        result = decision(entries, [('time:06:30:unspecified', .99), ('time:07:30:am', .1)],
                          'SET_ALARM', 'time', gate)
        self.assertEqual(result['status'], 'clarify')
        self.assertEqual(result['reason'], 'missing_meridiem')
        self.assertIsNone(result.get('value'))

    def test_no_fake_probability(self):
        result = decision(self.entries, [('number:75', .99), ('number:25', .1)],
                          'BRIGHTNESS', 'number', self.gate)
        self.assertEqual(result['value'], 75)
        self.assertFalse(result['confidence_is_probability'])
        self.assertFalse(result['action_executed'])

    def test_unknowns_count_against_calibration(self):
        rows = [dict(scores=[('a', .9), ('b', .1)], reference='a') for _ in range(10)]
        rows += [dict(scores=[('a', .95), ('b', .1)], reference=None) for _ in range(20)]
        self.assertFalse(calibrate(rows, 'v')['passed'])

    def test_homophone_alias_collision_rejects(self):
        entries = [dict(id='a', aliases=['adele'], role='artist'),
                   dict(id='b', aliases=['adell'], role='artist')]
        gate = dict(self.gate, catalog_version=catalog_fingerprint(entries))
        result = decision(entries, [('a', .9), ('b', .9)], 'MEDIA_PLAY', 'artist', gate)
        self.assertEqual(result['reason'], 'low_score_or_margin')


if __name__ == '__main__':
    unittest.main()
