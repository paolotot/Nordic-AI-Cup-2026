"""Our model: Parakeet ASR + a local Gemma 4 E4B answering all ten questions at once.

Pipeline per request (one conversation, ten questions):
  1. Parakeet TDT 0.6B (int8, CPU) transcribes the MP3 with word timings.
  2. The transcript is cut into numbered sentence lines.
  3. Gemma 4 E4B (llama.cpp on the Intel iGPU) reads the lines and all ten
     questions in one prompt and answers 'Qn: [cited lines] yes|no'.
  4. Each yes gets the cited lines' span, trimmed to the clauses that match
     the question.

Both models load and warm up at import time, before the attempt starts.
Everything runs on this machine; no cloud API is called.

Never raises: if the LLM fails or the time budget is nearly spent, a keyword
heuristic answers instead, and if even ASR fails we return a guess. A wrong
answer costs one mark; a failed request costs ten.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

import answering
import asr
import llm_server
from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import AUDIO_DIRECTORY, decode_audio

logger = logging.getLogger(__name__)

# The evaluator allows 60 s per request from the POST; keep a margin for
# network, base64 and JSON on both sides.
REQUEST_BUDGET_S = 55.0
MIN_LLM_S = 4.0  # below this there is no point starting the LLM call
# Stop ASR here (partial transcript) so the LLM keeps ~18 s even when the
# machine is running slow. Worst seen: ASR overshoots its deadline by ~4 s
# (it finishes the current VAD batch), answering takes up to ~18 s, so
# 33 + 4 + 18 = 55 s, plus network, under the evaluator's 60 s.
ASR_DEADLINE_S = 33.0


# --------------------------------------------------------------------------- #
# Load and warm up at import time
# --------------------------------------------------------------------------- #

def _load():
    t0 = time.perf_counter()
    asr_model = asr.load()
    llm_proc = llm_server.start()
    logger.info('Models loaded in %.1fs', time.perf_counter() - t0)

    # The first inference is the slowest; pay for it now rather than on the
    # first scored conversation.
    t0 = time.perf_counter()
    sample = next(iter(sorted(Path(AUDIO_DIRECTORY).glob('*.mp3'))), None)
    if sample is not None:
        segments = asr.transcribe(asr_model, sample.read_bytes())
    else:
        segments = [{'start': 0.0, 'end': 1.0, 'text': 'Hello.', 'words': []}]
    answering.answer_questions(segments, ['Is there a greeting?'], server=llm_server.URL)
    logger.info('Warm-up done in %.1fs', time.perf_counter() - t0)
    return asr_model, llm_proc


ASR_MODEL, LLM_PROC = _load()


# --------------------------------------------------------------------------- #
# The endpoint's model
# --------------------------------------------------------------------------- #

def _clean(span) -> Optional[tuple]:
    if span is None:
        return None
    start, end = round(float(span[0]), 2), round(float(span[1]), 2)
    return (start, end) if end > start >= 0 else None


def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    t0 = time.perf_counter()
    questions = request.questions
    n = len(questions)

    answers: List[bool] = [True] * n
    spans: List[Optional[tuple]] = [None] * n
    how = 'guess'
    t_asr = float('nan')

    try:
        segments = asr.transcribe(ASR_MODEL, decode_audio(request.audio_base64),
                                  deadline=t0 + ASR_DEADLINE_S)
        t_asr = time.perf_counter() - t0

        remaining = REQUEST_BUDGET_S - t_asr
        try:
            if remaining < MIN_LLM_S:
                raise TimeoutError(f'only {remaining:.1f}s left after ASR')
            answers, spans, _ = answering.answer_questions(
                segments, questions, server=llm_server.URL, timeout=remaining)
            how = 'llm'
        except Exception:
            logger.exception('LLM failed for %s, using keyword fallback',
                             request.audio_filename)
            answers, spans = answering.keyword_fallback(segments, questions)
            how = 'keyword'
    except Exception:
        logger.exception('ASR failed for %s, returning a guess', request.audio_filename)

    spans = [_clean(s) if a else None for a, s in zip(answers, spans)]
    t_total = time.perf_counter() - t0
    logger.info('%s: %d questions, %d yes, via %s, asr %.1fs + answer %.1fs = %.1fs total',
                request.audio_filename, n, sum(answers), how,
                t_asr, t_total - t_asr, t_total)

    return ASRQuestionResponseDto(
        answers=[bool(a) for a in answers],
        evidence_start=[s[0] if s else None for s in spans],
        evidence_end=[s[1] if s else None for s in spans],
    )
