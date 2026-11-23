"""Opt-in, line-delimited progress messages on the web worker's stdout."""

import json
import os
import time

PREFIX = "LWM_PROGRESS "


def compatible_digest(value):
    # Exact compatible revisions. Changes to fixed-count clustering use a separate diarization key.
    return {
        "9ed485c2586ee8dd40f5eb31371ed6aa9594cb0af29ffe1232e9834c71282c6f": "fa009543e5ad8f90e3fa6f9e8178a7ed52d2cbea1e8ba416d6c5e77ad5304de9",
        "c13cf5e335754a61148998ed6498b1de087c2a6f4aa43047dfcb991e48f2cce3": "572654cb84790d6bec3a2c3f7ccc194e466d76ec7d6f33e9891064a308ff6340",
        "a7341d12689699c9acc2414dc0381e9fb6f17418f8636d1b89f9483a7889c43f": "8c2b0cec3f33358760e1f7a832ba89c78de717ace9f4d1c4ad606c2c9985f68a",
    }.get(value, value)


class Reporter:
    def __init__(self, stage, unit):
        self.stage = stage
        self.unit = unit
        self.enabled = os.environ.get("LWM_WEB_PROGRESS") == "1"
        self.last = None

    def __call__(self, done=0, total=0):
        if not self.enabled:
            return
        now = time.monotonic()
        if self.last is not None and done < total and now - self.last < 0.4:
            return
        self.last = now
        print(PREFIX + json.dumps([self.stage, self.unit, done, total]), flush=True)
