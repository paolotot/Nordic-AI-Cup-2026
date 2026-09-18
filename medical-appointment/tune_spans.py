"""Compare evidence-span strategies offline, on the lines an LLM run already cited.

Reads the per-question records written by ``eval_answering.py`` (they carry the
model's raw 'Qn: [lines] yes|no' output), rebuilds the lines from the cached
transcript, and scores each strategy's tIoU against the annotations. No model
calls, so this runs in a second.

    python tune_spans.py outputs/answers-Qwen3.5-4B-Q4_K_M-parakeet-tdt-0.6b-v2-int8.json
"""

import argparse
import json
from pathlib import Path

from answering import parse_rows, span_from_lines, split_lines, trim_span
from utils import gold_evidence, temporal_iou

HERE = Path(__file__).resolve().parent


def minmax(lines, cited):
    cited = [i for i in cited if 0 <= i < len(lines)]
    return (lines[min(cited)]['start'], lines[max(cited)]['end']) if cited else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('records', type=Path)
    parser.add_argument('--transcripts', default='parakeet-tdt-0.6b-v2-int8')
    args = parser.parse_args()

    records = json.loads(args.records.read_text())
    by_conv = {}
    for r in records:
        by_conv.setdefault(r['transcript_id'], []).append(r)

    strategies = {
        'min-max of cited lines': lambda L, c, q: minmax(L, c),
        'largest contiguous block': lambda L, c, q: span_from_lines(L, c),
        'block + clause trim': lambda L, c, q: trim_span(L, c, q, mode='clause'),
        'block + word trim': lambda L, c, q: trim_span(L, c, q, mode='word'),
    }
    scores = {k: [] for k in strategies}
    for tid, rows in by_conv.items():
        segments = json.loads((HERE / 'transcripts' / args.transcripts / f'{tid}.json')
                              .read_text(encoding='utf-8'))
        lines = split_lines(segments)
        parsed = parse_rows(rows[0]['raw'] or '', len(rows))
        for row, (yes, cited) in zip(rows, parsed):
            gold = gold_evidence(row)
            if row['label'] != '1' or not gold:
                continue
            for name, fn in strategies.items():
                scores[name].append(temporal_iou(gold, fn(lines, cited, row['question']) if yes else None))

    n = len(next(iter(scores.values())))
    print(f'mean tIoU over {n} annotated yes questions')
    for name, s in scores.items():
        print(f'  {name:<28} {sum(s) / n:.3f}')


if __name__ == '__main__':
    main()
