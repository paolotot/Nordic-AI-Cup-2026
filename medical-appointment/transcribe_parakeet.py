"""Transcribe every training conversation with NVIDIA Parakeet and time it.

Same output shape as ``transcribe_all.py`` (segments with word timings) so the
two can be compared file by file, written to ``transcripts/<model>/``.

    python transcribe_parakeet.py
    python transcribe_parakeet.py --limit 5 --quantization int8
"""

import argparse
import json
import time
from pathlib import Path

from asr import load, transcribe
from utils import AUDIO_DIRECTORY

OUT_DIRECTORY = Path(__file__).resolve().parent / 'transcripts'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantization', default='int8')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    name = f'parakeet-tdt-0.6b-v2-{args.quantization or "fp32"}'
    out = OUT_DIRECTORY / name
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    asr = load(args.quantization)
    print(f'[{name}] loaded in {time.perf_counter() - t0:.1f}s', flush=True)

    files = sorted(AUDIO_DIRECTORY.glob('*.mp3'),
                   key=lambda p: int(p.stem.rsplit('_', 1)[1]))[:args.limit]
    timings = []
    for path in files:
        t0 = time.perf_counter()
        segments = transcribe(asr, path)
        elapsed = time.perf_counter() - t0
        audio_s = segments[-1]['end'] if segments else 0
        transcript_id = path.stem.removeprefix('conversation_')
        (out / f'{transcript_id}.json').write_text(
            json.dumps(segments, indent=1), encoding='utf-8')
        timings.append({'id': transcript_id, 'asr_s': elapsed})
        print(f'[{name}] {transcript_id:>10}  speech to {audio_s:6.1f}s  '
              f'asr {elapsed:5.1f}s', flush=True)

    asr_s = [t['asr_s'] for t in timings]
    summary = {'model': name, 'mean_asr_s': sum(asr_s) / len(asr_s),
               'max_asr_s': max(asr_s), 'files': timings}
    (out / '_timing.json').write_text(json.dumps(summary, indent=1))
    print(f'[{name}] mean {summary["mean_asr_s"]:.1f}s  '
          f'max {summary["max_asr_s"]:.1f}s per conversation', flush=True)
