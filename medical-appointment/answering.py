"""The answering half: transcript + ten questions -> answers and evidence spans.

The transcript is split into short numbered lines (sentences, from the word
timings). The local LLM gets every line and every question in one prompt and
returns, per question, the line numbers it read the answer off and a yes/no.
The cited lines' start/end times become the evidence span.

The LLM runs in a local llama.cpp server (``llama-server``) on this machine,
so nothing leaves the box at inference time.
"""

import re
from typing import List, Optional, Tuple

import requests

Span = Tuple[float, float]

SENTENCE_END = re.compile(r'[.?!]["\')]?$')
CLAUSE_END = re.compile(r'[,;:]["\')]?$')
PAUSE_SPLIT_S = 1.0  # also break a line at a silence this long

# 'sentence': break at . ? ! and long pauses.
# 'clause':   also break at , ; : once the line has CLAUSE_MIN_WORDS words,
#             so the LLM can cite part of a sentence.
# 'pause':    also break at any silence >= PAUSE_LINE_S; the annotated spans
#             start and end on pauses (the audio is stitched TTS clips).
LINE_SPLIT = 'sentence'
PAUSE_LINE_S = 0.3
TRIM_MODE = 'pause'  # how trim_span cuts inside cited lines
CLAUSE_MIN_WORDS = 3


def split_lines(segments: List[dict], mode: Optional[str] = None,
                clause_min_words: Optional[int] = None) -> List[dict]:
    """Turn ASR segments (with word timings) into short numbered lines."""
    mode = mode or LINE_SPLIT
    min_words = CLAUSE_MIN_WORDS if clause_min_words is None else clause_min_words
    lines = []
    for seg in segments:
        words = seg.get('words') or []
        if not words:
            lines.append({'start': seg['start'], 'end': seg['end'],
                          'text': seg['text'].strip(), 'words': []})
            continue
        current = []
        for i, w in enumerate(words):
            current.append(w)
            nxt = words[i + 1] if i + 1 < len(words) else None
            gap = (nxt['start'] - w['end']) if nxt else 0
            token = w['word'].strip()
            clause_break = ((mode == 'clause' and CLAUSE_END.search(token)
                             and len(current) >= min_words)
                            or (mode == 'pause' and gap >= PAUSE_LINE_S))
            if nxt is None or SENTENCE_END.search(token) or clause_break \
                    or gap >= PAUSE_SPLIT_S:
                lines.append({
                    'start': current[0]['start'], 'end': current[-1]['end'],
                    'text': ''.join(x['word'] for x in current).strip(),
                    'words': current,
                })
                current = []
    return [l for l in lines if l['text']]


CITE_RULES = {
    # v1: what scored 0.745 locally / 0.712 on validation
    'minimal': """1. Find the transcript lines that the question is about. Cite the smallest set of consecutive lines that contains the answer - usually one to three lines. Do not cite lines that are merely on the same general topic.""",
    # v2: annotations often cover a whole exchange, not just the answer line
    'exchange': """1. Find the passage that establishes the answer and cite all of its consecutive lines. When the fact comes out of an exchange - a question and the reply to it, or an examination followed by its finding - cite the whole exchange, from the line that raises it to the line that settles it. Do not add lines that only acknowledge or repeat it ("Yes.", "Okay.", a number said again), and do not cite lines that are merely on the same general topic.""",
}
CITE_RULE = 'minimal'

SYSTEM_PROMPT_TEMPLATE = """You check yes/no questions against the transcript of a recorded doctor-patient consultation.

For each question:
{cite_rule}
2. Answer true ONLY if those lines establish exactly what the question states. Check every detail: drug name, dose, number, unit, frequency, duration, body side and location, test result, who said or did what, and whether something was actually agreed or only mentioned.
3. Answer false if any detail differs (for example 200 mg when the transcript says 100 mg, six weeks when it says two weeks, left knee when it says right knee), if it was never mentioned, or if it contradicts the transcript. For a false answer, cite the lines that show the conflict, or none if the subject never comes up.

The transcript is automatic speech recognition, so drug names and surnames may be misspelled; treat close-sounding spellings as the same word. The question's phrasing (plain question, "right?", "didn't it?") tells you nothing about the answer."""


# Invented examples (not from the training data) showing how the annotated
# evidence is cut: whole lines; the full exchange when the answer alone is
# meaningless; no trailing repetitions or acknowledgements.
FEW_SHOT = """

Examples of correct citations:

[40] How are the headaches since we changed the tablets?
[41] Much better.
[42] I have only had two this month.
[43] Good, that is what we hoped for.
Q: Have the headaches improved since the medication change? -> [40,41,42] yes
(The reply "Much better." only means something together with the question it answers, so the question is cited too.)

[12] Your pulse is 72 and regular.
[13] 72, okay.
[14] And your temperature is normal.
Q: Was the pulse recorded as 72? -> [12] yes
(Line 13 only repeats it, so it is not cited.)

[25] Let me have a look at your ears.
[26] The left one is a little red, but the right one is fine.
Q: Was the right ear found to be normal? -> [26] yes
Q: Was the left ear found to be normal? -> [26] no"""
USE_FEW_SHOT = False
QUESTIONS_FIRST = False


def system_prompt(cite_rule: str = None) -> str:
    prompt = SYSTEM_PROMPT_TEMPLATE.format(cite_rule=CITE_RULES[cite_rule or CITE_RULE])
    return prompt + (FEW_SHOT if USE_FEW_SHOT else '')


def build_prompt(lines: List[dict], questions: List[str]) -> str:
    transcript = '\n'.join(f'[{i}] {l["text"]}' for i, l in enumerate(lines))
    qs = '\n'.join(f'Q{i + 1}. {q}' for i, q in enumerate(questions))
    body = (f'QUESTIONS\n{qs}\n\nTRANSCRIPT\n{transcript}\n\n' if QUESTIONS_FIRST
            else f'TRANSCRIPT\n{transcript}\n\nQUESTIONS\n{qs}\n\n')
    return (body +
            f'Answer all {len(questions)} questions in order, one per line, in the form\n'
            f'Q<n>: [<cited line numbers, comma separated>] yes|no\n'
            f'for example "Q1: [12,13] yes" or "Q2: [] no".')


def response_grammar(n_questions: int) -> str:
    """GBNF forcing exactly one 'Qn: [a,b] yes|no' line per question.

    Far fewer output tokens than JSON, and generation is the slow part on CPU.
    """
    rows = ' '.join(f'"Q{i + 1}: " cited " " verdict "\\n"' for i in range(n_questions))
    return (f'root ::= {rows}\n'
            'cited ::= "[" ( num ( "," num ){0,5} )? "]"\n'
            'num ::= [0-9] [0-9]? [0-9]?\n'
            'verdict ::= "yes" | "no"\n')


ROW = re.compile(r'^Q(\d+): \[([0-9,]*)\] (yes|no)$', re.M)


def parse_rows(raw: str, n_questions: int) -> List[Tuple[bool, List[int]]]:
    parsed = {int(m[1]): (m[3] == 'yes', [int(x) for x in m[2].split(',') if x])
              for m in ROW.finditer(raw)}
    # Anything missing becomes a guess rather than a malformed response.
    return [parsed.get(i + 1, (True, [])) for i in range(n_questions)]


def contiguous_block(cited: List[int], max_gap: int = 1) -> List[int]:
    """The run of cited lines (allowing ``max_gap`` skipped lines) with the most citations.

    A model that cites line 3 and line 20 would otherwise produce a span over
    everything in between, which scores next to nothing.
    """
    cited = sorted(set(cited))
    if not cited:
        return []
    blocks, block = [], [cited[0]]
    for i in cited[1:]:
        if i - block[-1] <= max_gap + 1:
            block.append(i)
        else:
            blocks.append(block)
            block = [i]
    blocks.append(block)
    return max(blocks, key=len)  # ties -> earliest


STOPWORDS = set('''
a an the is are was were be been being am do does did done have has had having
to of in on at for with by from as and or but if so than then that this these those
it its i you he she we they me my your his her our their them him us
there here what which who whom whose when where why how not no yes any some all
will would should could can may might must shall right correct isnt wasnt didnt doesnt
arent werent patient doctor mention mentioned discussed discussion conversation
'''.split())

NUMBER_WORDS = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
                'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
                'twelve': '12', 'fifteen': '15', 'twenty': '20', 'thirty': '30',
                'once': '1', 'twice': '2', 'mg': 'milligrams'}


def _norm(word: str) -> str:
    w = re.sub(r"[^a-z0-9]", '', word.lower())
    return NUMBER_WORDS.get(w, w)


def _content(text: str) -> List[str]:
    return [w for w in (_norm(x) for x in text.split()) if w and w not in STOPWORDS]


def _matches(word: str, keys: List[str]) -> bool:
    """Same word, same stem (5-char prefix), or a close spelling (ASR drug names)."""
    import difflib
    for k in keys:
        if word == k or (len(word) >= 5 and len(k) >= 5 and word[:5] == k[:5]):
            return True
        if len(word) >= 5 and difflib.SequenceMatcher(None, word, k).ratio() >= 0.8:
            return True
    return False


PAUSE_CHUNK_S = 0.4


def trim_span(lines: List[dict], cited: List[int], question: str,
              mode: str = 'clause') -> Optional[Span]:
    """Evidence span over the cited block, trimmed to the part that matches the question.

    'clause': split the block at commas/sentence ends, keep the first to the
    last clause that shares a content word with the question.
    'word': keep the first to the last matching word.
    Falls back to the whole block when nothing matches.
    """
    block = contiguous_block([i for i in cited if 0 <= i < len(lines)])
    if not block:
        return None
    whole = (lines[block[0]]['start'], lines[block[-1]]['end'])
    words = [w for i in block for w in lines[i]['words']]
    keys = _content(question)
    if not words or not keys:
        return whole

    hits = [i for i, w in enumerate(words) if (n := _norm(w['word'])) and
            n not in STOPWORDS and _matches(n, keys)]
    if not hits:
        return whole

    if mode == 'word':
        return words[hits[0]]['start'], words[hits[-1]]['end']

    # 'clause': a chunk ends at a word carrying , . ? ! ;
    # 'pause':  a chunk ends at a silence >= PAUSE_CHUNK_S (the annotated
    #           spans start and end on pauses: median 0.78 s of silence
    #           before a gold start, vs 0.08 s between ordinary words)
    clause_of, c = [], 0
    for k, w in enumerate(words):
        clause_of.append(c)
        if mode == 'pause':
            nxt = words[k + 1] if k + 1 < len(words) else None
            if nxt is not None and nxt['start'] - w['end'] >= PAUSE_CHUNK_S:
                c += 1
        elif re.search(r'[,.?!;]$', w['word'].strip()):
            c += 1
    keep = {clause_of[i] for i in hits}
    first = next(i for i, w in enumerate(words) if clause_of[i] == min(keep))
    last = max(i for i, w in enumerate(words) if clause_of[i] == max(keep))
    return words[first]['start'], words[last]['end']


# Annotators start a touch before the first word and end a touch after the
# last; fitted on the 195 training spans (tIoU 0.590 -> 0.598).
PAD_START_S = 0.10
PAD_END_S = 0.05


def pad(span: Optional[Span]) -> Optional[Span]:
    if span is None:
        return None
    return max(0.0, span[0] - PAD_START_S), span[1] + PAD_END_S


REFINE_SYSTEM = """You mark evidence in the transcript of a recorded doctor-patient consultation.

You get a short excerpt of numbered transcript lines and a statement that the consultation supports. Give the consecutive lines that an annotator would highlight as the evidence for it: the lines that actually state or establish it. If it is established by an exchange (a question and its answer, or an examination and its finding), include both parts. Leave out lines before or after that only add other details, small talk, or acknowledgements like "Yes." or "Okay."

Reply with just the first and last line number, as "first-last" (the same number twice for a single line)."""

REFINE_WINDOW = 3  # lines of context on each side of the first-pass block


def refine_grammar() -> str:
    return 'root ::= num "-" num\nnum ::= [0-9] [0-9]? [0-9]?\n'


def refine_block(lines: List[dict], block: List[int], question: str,
                 server: str, timeout: float) -> List[int]:
    """Second, focused look at one yes-question: a small window, one question.

    Returns the chosen line indices (global), or the original block if the
    answer is unusable.
    """
    lo = max(0, block[0] - REFINE_WINDOW)
    hi = min(len(lines) - 1, block[-1] + REFINE_WINDOW)
    excerpt = '\n'.join(f'[{i}] {lines[i]["text"]}' for i in range(lo, hi + 1))
    body = {
        'messages': [
            {'role': 'system', 'content': REFINE_SYSTEM},
            {'role': 'user', 'content': f'EXCERPT\n{excerpt}\n\nSTATEMENT\n{question}'},
        ],
        'temperature': 0,
        'max_tokens': 12,
        'grammar': refine_grammar(),
        'chat_template_kwargs': {'enable_thinking': False},
    }
    r = requests.post(f'{server}/v1/chat/completions', json=body, timeout=timeout)
    r.raise_for_status()
    m = re.match(r'(\d+)-(\d+)', r.json()['choices'][0]['message']['content'])
    if not m:
        return block
    a, b = sorted((int(m[1]), int(m[2])))
    if a < lo or b > hi:
        return block
    return list(range(a, b + 1))


def keyword_fallback(segments: List[dict], questions: List[str]
                     ) -> Tuple[List[bool], List[Optional[Span]]]:
    """No-LLM guess, for when the model is down or out of time.

    Yes if some line (or pair of adjacent lines) covers most of the question's
    content words, with that line as the evidence. Cannot see hard negatives,
    but beats a constant answer on off-topic questions and still finds spans.
    """
    lines = split_lines(segments)
    answers, spans = [], []
    for q in questions:
        keys = _content(q)
        best, best_i = 0.0, None
        for i in range(len(lines)):
            text = ' '.join(l['text'] for l in lines[i:i + 2])
            words = {w for w in _content(text)}
            if keys:
                cover = sum(_matches(k, list(words)) for k in keys) / len(keys)
                if cover > best:
                    best, best_i = cover, i
        yes = best >= 0.5 and best_i is not None
        answers.append(yes)
        spans.append(trim_span(lines, [best_i], q) if yes else None)
    return answers, spans


def span_from_lines(lines: List[dict], cited: List[int]) -> Optional[Span]:
    block = contiguous_block([i for i in cited if 0 <= i < len(lines)])
    if not block:
        return None
    return lines[block[0]]['start'], lines[block[-1]]['end']


def answer_questions(
    segments: List[dict],
    questions: List[str],
    server: str = 'http://127.0.0.1:8080',
    timeout: float = 45,
    cite_rule: Optional[str] = None,
) -> Tuple[List[bool], List[Optional[Span]], dict]:
    """Ask the local LLM every question in one call.

    Returns answers, spans (None for a no), and debug info (raw output, token
    counts). Raises on server/parse failure; the caller decides the fallback.
    """
    lines = split_lines(segments)
    body = {
        'messages': [
            {'role': 'system', 'content': system_prompt(cite_rule)},
            {'role': 'user', 'content': build_prompt(lines, questions)},
        ],
        'temperature': 0,
        'max_tokens': 30 * len(questions),
        'grammar': response_grammar(len(questions)),
        'chat_template_kwargs': {'enable_thinking': False},
    }
    r = requests.post(f'{server}/v1/chat/completions', json=body, timeout=timeout)
    r.raise_for_status()
    out = r.json()
    raw = out['choices'][0]['message']['content']

    answers, spans = [], []
    for question, (yes, cited) in zip(questions, parse_rows(raw, len(questions))):
        answers.append(yes)
        spans.append(pad(trim_span(lines, cited, question, mode=TRIM_MODE)) if yes else None)

    debug = {'raw': raw, 'lines': lines, 'usage': out.get('usage'),
             'timings': out.get('timings')}
    return answers, spans, debug
