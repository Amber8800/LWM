import unittest
from types import SimpleNamespace

import numpy as np

from lwm.audio import RATE
from lwm.speech import decode_windows


class WindowTests(unittest.TestCase):
    def test_each_window_owns_each_spoken_marker_once(self):
        # Two short markers straddle candidate seams in a recording over 30 s.
        audio = np.zeros(58 * RATE, dtype=np.float32)
        for pos in (2, 25, 51):
            audio[pos * RATE : (pos * RATE) + 3200] = 0.3
        lengths = []

        class Decoder:
            def transcribe(self, clip, **settings):
                self_clip = np.r_[False, clip != 0, False]
                edges = np.flatnonzero(np.diff(self_clip))
                lengths.append(len(clip))
                self_words = [
                    SimpleNamespace(start=a / RATE, end=b / RATE, word=" 발화")
                    for a, b in zip(edges[::2], edges[1::2])
                ]
                return [SimpleNamespace(words=self_words)], None

        words = decode_windows(Decoder(), audio, RATE, None)
        self.assertEqual([w["start_ms"] for w in words], [3000, 26000, 52000])
        self.assertLessEqual(max(lengths), 30 * RATE)
        self.assertEqual(decode_windows(Decoder(), audio[:RATE], 0, None), [])
