import unittest

from lwm.nemotron import Segment, turns_from_segments
from lwm.web import stage_progress


class NemotronTests(unittest.TestCase):
    def test_overlapping_channels_and_padded_tail(self):
        turns = turns_from_segments([Segment(0.4, 1.3, 2), Segment(0.1, 0.8, 1)], 1000)
        self.assertEqual(
            [(t["start_ms"], t["end_ms"], t["speaker"], t["overlap"]) for t in turns],
            [
                (100, 400, "SPEAKER_01", False),
                (400, 800, "SPEAKER_01", True),
                (400, 800, "SPEAKER_02", True),
                (800, 1000, "SPEAKER_02", False),
            ],
        )
        self.assertEqual(turns_from_segments([], 1000), [])

    def test_progress_uses_only_active_models(self):
        for diarizer, expected in [("nemotron", 50), ("community", 66)]:
            self.assertEqual(
                stage_progress("txt", "asr", "load", 0, 0, diarizer)["percent"],
                expected,
            )
