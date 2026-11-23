import sys
from pathlib import Path

from .common import offline, read, save


def main():
    offline()
    stage, folder = sys.argv[1], Path(sys.argv[2])
    from .minutes import summarize
    from .nemotron import diarize as nemotron
    from .speech import diarize, segment, transcribe

    func = {
        "asr": transcribe,
        "nemotron": nemotron,
        "segment": segment,
        "diarize": diarize,
        "minutes": summarize,
    }[stage]
    save(folder / "result.json", func(read(folder / "job.json")))


if __name__ == "__main__":
    main()
