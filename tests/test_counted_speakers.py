import unittest

import numpy as np

from lwm.speech import counted_labels


class CountedSpeakersTests(unittest.TestCase):
    def test_recurring_voices_take_priority_over_an_isolated_outlier(self):
        a, b, noise = np.eye(3)
        vectors = [noise, a, b, a, b, a, b]
        records = [{"window": i} for i in range(len(vectors))]
        labels = counted_labels(records, list(range(len(vectors))), vectors, 2, 0.45)
        self.assertNotIn(0, labels)
        self.assertEqual(labels[1], labels[5])
        self.assertEqual(labels[2], labels[6])
        self.assertNotEqual(labels[1], labels[2])

    def test_two_slots_in_one_window_cannot_claim_the_same_voice(self):
        records = [{"window": w} for w in (0, 1, 1)]
        labels = counted_labels(records, [0, 1, 2], [[1.0, 0.0]] * 3, 1, 0.45)
        self.assertEqual(len(labels), 2)
        self.assertEqual(set(labels.values()), {"SPEAKER_01"})
