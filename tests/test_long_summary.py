import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from lwm.common import save
from lwm.minutes import summarize


class LongSummaryTest(unittest.TestCase):
    def test_multiple_reduction_levels_preserve_late_revision_and_sources(self):
        rows = [
            {
                "id": f"U{i:03}",
                "speaker": "A",
                "overlap": False,
                "text": f"일정 {i}차 수정",
            }
            for i in range(15)
        ]
        calls = []

        class BudgetedServer:
            def tokens(self, prompt):
                return 800 * len(json.loads(prompt.splitlines()[-1]))

            def generate(self, prompt, limit, sources):
                group = json.loads(prompt.splitlines()[-1])
                calls.append(len(group))
                if len(group) > 3:
                    raise AssertionError("context exceeded")
                latest = group[-1]
                if "evidence" in latest:
                    return [latest]
                return [
                    {
                        "kind": "결정 사항",
                        "text": latest["text"],
                        "owner": "미정",
                        "due": "미정",
                        "evidence": [{"id": latest["id"], "quote": latest["text"]}],
                    }
                ]

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "t.json"
            save(path, {"utterances": rows})
            with patch(
                "lwm.minutes.server", return_value=nullcontext(BudgetedServer())
            ):
                job = {"transcript": str(path), "cache": root, "context": 4096}
                result = summarize(job)
                self.assertEqual(result[0]["evidence"][0]["id"], "U014")
                self.assertEqual(len(calls), 8)
                self.assertEqual(summarize(job), result)
                self.assertEqual(len(calls), 8)
