"""Measure the second-pass span refinement offline.

Takes the first-pass citations from an eval_answering.py records file, runs
``answering.refine_block`` for every question the first pass answered yes,
and compares tIoU before and after. Needs a llama-server on :8080 (the live
server's is fine).

    python tune_refine.py outputs/answers-...-minimal.json --window 3
"""

import argparse
import collections
import json
import time
from pathlib import Path

import answering
from answering import contiguous_block, pad, parse_rows, refine_block, split_lines, trim_span
from utils import gold_evidence, temporal_iou

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('records', type=Path)
    parser.add_argument('--transcripts', default='parakeet-tdt-0.6b-v2-int8')
    parser.add_argument('--window', type=int, default=3)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    answering.REFINE_WINDOW = args.window

    conv = collections.defaultdict(list)
    for r in json.loads(args.records.read_text()):
        conv[r['transcript_id']].append(r)

    before, after, after_notrim, per_conv_s = [], [], [], []
    for tid, rows in list(conv.items())[:args.limit]:
        segments = json.loads((HERE / 'transcripts' / args.transcripts / f'{tid}.json')
                              .read_text(encoding='utf-8'))
        lines = split_lines(segments)
        t0 = time.perf_counter()
        for row, (yes, cited) in zip(rows, parse_rows(rows[0]['raw'] or '', len(rows))):
            block = contiguous_block([i for i in cited if 0 <= i < len(lines)])
            refined = block
            if yes and block:
                refined = refine_block(lines, block, row['question'],
                                       'http://127.0.0.1:8080', timeout=30)
            gold = gold_evidence(row)
            if row['label'] != '1' or not gold:
                continue
            q = row['question']
            before.append(temporal_iou(gold, pad(trim_span(lines, block, q))) if yes else 0)
            after.append(temporal_iou(gold, pad(trim_span(lines, refined, q))) if yes else 0)
            after_notrim.append(temporal_iou(gold, pad((lines[refined[0]]['start'], lines[refined[-1]]['end'])))
                                if yes and refined else 0)
        per_conv_s.append(time.perf_counter() - t0)
        print(f'{tid:>10}  refine {per_conv_s[-1]:4.1f}s', flush=True)

    n = len(before)
    print(f'\nwindow {args.window}, {n} annotated yes questions')
    print(f'  first pass            {sum(before) / n:.3f}')
    print(f'  refined + clause trim {sum(after) / n:.3f}')
    print(f'  refined, no trim      {sum(after_notrim) / n:.3f}')
    print(f'  refine time/conv      mean {sum(per_conv_s) / len(per_conv_s):.1f}s  max {max(per_conv_s):.1f}s')


if __name__ == '__main__':
    main()
