"""Transcribe every training conversation and time it.

Writes one JSON per conversation to ``transcripts/<model>/`` with segment and
word timings, plus ``transcripts/<model>/_timing.json``. Those cached
transcripts are what we tune the answering half against, so we do not pay for
ASR on every experiment.

    python transcribe_all.py                          # default models
    python transcribe_all.py --models small.en
"""

import argparse
import json
import os
import time
from pathlib import Path

from faster_whisper import WhisperModel

from utils import AUDIO_DIRECTORY, audio_duration_seconds

OUT_DIRECTORY = Path(__file__).resolve().parent / 'transcripts'


def transcribe(model: WhisperModel, path: Path) -> list[dict]:
    segments, _ = model.transcribe(
        str(path),
        language='en',
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
    )
    return [
        {
            'start': s.start,
            'end': s.end,
            'text': s.text.strip(),
            'words': [
                {'start': w.start, 'end': w.end, 'word': w.word}
                for w in (s.words or [])
            ],
        }
        for s in segments
    ]


def run(model_name: str, threads: int) -> None:
    out = OUT_DIRECTORY / model_name
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    model = WhisperModel(model_name, device='cpu', compute_type='int8',
                         cpu_threads=threads)
    load_s = time.perf_counter() - t0
    print(f'[{model_name}] loaded in {load_s:.1f}s', flush=True)

    files = sorted(AUDIO_DIRECTORY.glob('*.mp3'),
                   key=lambda p: int(p.stem.rsplit('_', 1)[1]))
    timings = []
    for path in files:
        duration = audio_duration_seconds(path.read_bytes())
        t0 = time.perf_counter()
        segments = transcribe(model, path)
        elapsed = time.perf_counter() - t0

        transcript_id = path.stem.removeprefix('conversation_')
        (out / f'{transcript_id}.json').write_text(
            json.dumps(segments, indent=1), encoding='utf-8')
        timings.append({'id': transcript_id, 'audio_s': duration,
                        'asr_s': elapsed})
        print(f'[{model_name}] {transcript_id:>10}  audio {duration:6.1f}s  '
              f'asr {elapsed:5.1f}s', flush=True)

    asr = [t['asr_s'] for t in timings]
    summary = {'model': model_name, 'load_s': load_s,
               'mean_asr_s': sum(asr) / len(asr), 'max_asr_s': max(asr),
               'files': timings}
    (out / '_timing.json').write_text(json.dumps(summary, indent=1))
    print(f'[{model_name}] mean {summary["mean_asr_s"]:.1f}s  '
          f'max {summary["max_asr_s"]:.1f}s per conversation', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['small.en', 'distil-large-v3'])
    parser.add_argument('--threads', type=int, default=os.cpu_count() or 8)
    args = parser.parse_args()
    for name in args.models:
        run(name, args.threads)
