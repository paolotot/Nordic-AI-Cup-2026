"""Score the answering half on cached transcripts, without any ASR.

Starts llama-server with the given GGUF, runs every training conversation
through ``answering.answer_questions`` and prints accuracy by question type,
mean tIoU, the final score, and seconds per conversation. Also prints the
line-split ceiling: the best tIoU any choice of consecutive lines could get,
i.e. how much the evidence half is capped by our line boundaries.

    python eval_answering.py models/Qwen3.5-4B-Q4_K_M.gguf
    python eval_answering.py models/gemma-4-E4B-it-Q4_K_M.gguf --limit 5
"""

import argparse
import collections
import json
import time
from pathlib import Path

import answering
import llm_server
from answering import answer_questions, split_lines
from utils import gold_evidence, group_questions_by_conversation, temporal_iou

HERE = Path(__file__).resolve().parent


def line_ceiling(lines, gold):
    """Best tIoU achievable by citing any run of consecutive lines."""
    best = 0.0
    for i in range(len(lines)):
        for j in range(i, min(i + 8, len(lines))):
            best = max(best, temporal_iou(gold, (lines[i]['start'], lines[j]['end'])))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--transcripts', default='parakeet-tdt-0.6b-v2-int8')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--ctx', type=int, default=8192)
    parser.add_argument('--cpu', action='store_true', help='CPU build instead of the iGPU (Vulkan)')
    parser.add_argument('--cite-rule', default=None, help='key of answering.CITE_RULES')
    parser.add_argument('--split', default=None, help="answering.LINE_SPLIT: 'sentence' or 'clause'")
    parser.add_argument('--tag', default='', help='suffix for the output file name')
    parser.add_argument('--questions-first', action='store_true')
    parser.add_argument('--few-shot', action='store_true')
    args = parser.parse_args()
    if args.split:
        answering.LINE_SPLIT = args.split
    answering.QUESTIONS_FIRST = args.questions_first
    answering.USE_FEW_SHOT = args.few_shot

    (HERE / 'outputs').mkdir(exist_ok=True)
    tdir = HERE / 'transcripts' / args.transcripts
    groups = group_questions_by_conversation()[:args.limit]

    proc = llm_server.start(
        args.model, ctx=args.ctx, threads=args.threads,
        server=HERE / 'models' / ('llama-cpp' if args.cpu else 'llama-cpp-vulkan') / 'llama-server.exe',
        ngl=0 if args.cpu else 99)
    try:
        # Warm-up, as the real server would do at import time.
        answer_questions([{'start': 0, 'end': 1, 'text': 'Hello.', 'words': []}],
                         ['Is this a greeting?'])

        by_type = collections.defaultdict(lambda: [0, 0])
        ious, ceilings, times, records = [], [], [], []
        for _, rows in groups:
            transcript_id = rows[0]['transcript_id']
            segments = json.loads((tdir / f'{transcript_id}.json').read_text(encoding='utf-8'))
            questions = [r['question'] for r in rows]
            t0 = time.perf_counter()
            try:
                answers, spans, debug = answer_questions(segments, questions, timeout=120,
                                                         cite_rule=args.cite_rule)
            except Exception as e:  # scored as the fallback would be
                print(f'{transcript_id}: FAILED {e!r}')
                answers, spans, debug = [True] * len(rows), [None] * len(rows), {}
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            lines = split_lines(segments)

            for row, ans, span in zip(rows, answers, spans):
                ok = ans == (row['label'] == '1')
                by_type[row['question_type']][0] += ok
                by_type[row['question_type']][1] += 1
                gold = gold_evidence(row)
                iou = None
                if row['label'] == '1' and gold:
                    iou = temporal_iou(gold, span)
                    ious.append(iou)
                    ceilings.append(line_ceiling(lines, gold))
                records.append({**row, 'pred': ans, 'span': span, 'iou': iou,
                                'raw': debug.get('raw')})

            usage = debug.get('usage') or {}
            print(f'{transcript_id:>10}  {elapsed:5.1f}s  prompt {usage.get("prompt_tokens")} '
                  f'out {usage.get("completion_tokens")}  '
                  f'acc {sum(a == (r["label"] == "1") for a, r in zip(answers, rows))}/{len(rows)}',
                  flush=True)
    finally:
        proc.terminate()

    correct = sum(c for c, _ in by_type.values())
    total = sum(n for _, n in by_type.values())
    acc = correct / total
    miou = sum(ious) / len(ious)
    print(f'\n== {args.model.name} on {args.transcripts}, {len(groups)} conversations')
    for t, (c, n) in sorted(by_type.items()):
        print(f'  {t:<14} {c / n:.3f}  ({c}/{n})')
    print(f'  accuracy       {acc:.3f}')
    print(f'  mean tIoU      {miou:.3f}   (line-split ceiling {sum(ceilings) / len(ceilings):.3f})')
    print(f'  SCORE          {0.4 * acc + 0.6 * miou:.3f}')
    print(f'  time/conv      mean {sum(times) / len(times):.1f}s  max {max(times):.1f}s')

    out = HERE / 'outputs' / f'answers-{args.model.stem}-{args.transcripts}-{args.cite_rule or "default"}{args.tag}.json'
    out.write_text(json.dumps(records, indent=1))
    print(f'  per-question records: {out.relative_to(HERE)}')


if __name__ == '__main__':
    main()
