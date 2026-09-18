"""Speech recognition: NVIDIA Parakeet TDT 0.6B v2 (int8 ONNX) on CPU.

Returns segments with word-level timings, which the answering half turns into
evidence spans. Runs fully locally via onnx-asr; the weights are downloaded
from Hugging Face once and cached.
"""

import io
import logging
import time
from typing import List, Optional, Union

import onnx_asr
from faster_whisper.audio import decode_audio

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
FRAME_S = 0.08  # Parakeet emits one encoder frame every 80 ms


def load(quantization: str = 'int8'):
    model = onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v2', quantization=quantization)
    vad = onnx_asr.load_vad('silero')
    return model.with_vad(vad, max_speech_duration_s=20,
                          min_silence_duration_ms=300).with_timestamps()


def words_from_tokens(tokens, timestamps, offset, seg_end):
    """Merge subword tokens into words. A token starting with a space opens a word.

    Timestamps are token *starts*, so a word ends one frame after its last
    token, capped at the next word's start (and the segment end).
    Punctuation tokens are emitted after a pause, often well after the word
    was spoken, so they join the word's text but not its timing.
    """
    words = []
    for token, ts in zip(tokens, timestamps):
        if token.startswith((' ', '▁')) or not words:
            words.append({'start': offset + ts, 'last': offset + ts,
                          'word': token.replace('▁', ' ')})
        else:
            words[-1]['word'] += token
            if any(ch.isalnum() for ch in token):
                words[-1]['last'] = offset + ts
    for i, w in enumerate(words):
        limit = words[i + 1]['start'] if i + 1 < len(words) else seg_end
        w['end'] = min(w.pop('last') + FRAME_S, limit)
    return words


def transcribe(model, audio: Union[str, bytes], deadline: Optional[float] = None) -> List[dict]:
    """Transcribe a file path or raw MP3 bytes into timed segments.

    ``deadline`` is a ``time.perf_counter()`` value. Speech chunks come out of
    the VAD a batch at a time, so past the deadline we stop and return what we
    have: a partial transcript still answers most questions, whereas running
    over leaves the LLM no time at all.
    """
    source = io.BytesIO(audio) if isinstance(audio, bytes) else str(audio)
    waveform = decode_audio(source, sampling_rate=SAMPLE_RATE)
    segments = []
    for seg in model.recognize(waveform, sample_rate=SAMPLE_RATE):
        if deadline is not None and time.perf_counter() > deadline:
            logger.warning('ASR deadline hit at %.1fs of %.1fs audio',
                           seg.start, len(waveform) / SAMPLE_RATE)
            break
        if not seg.text.strip():
            continue
        segments.append({
            'start': seg.start, 'end': seg.end, 'text': seg.text.strip(),
            'words': words_from_tokens(seg.tokens or [], seg.timestamps or [],
                                       seg.start, seg.end),
        })
    return segments
