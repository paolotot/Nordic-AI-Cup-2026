# Our medical-appointment solution

Best validation score: **0.7697** (accuracy ~0.99, mean tIoU ~0.61), run from
Paolo's laptop (i7-13700H, Intel Iris Xe, no NVIDIA GPU). The organisers' spec
is in [README.md](README.md); this file is how to run and change ours.

## How it works

One request = one conversation + 10 questions. [example.py](example.py) does:

1. **Speech to text**: NVIDIA Parakeet TDT 0.6B v2, int8 ONNX
   ([asr.py](asr.py)). Gives word-level timestamps. Downloads itself from
   Hugging Face on first run (~700 MB).
2. **Lines**: the words are cut into numbered sentence lines.
3. **Answering**: Gemma 4 E4B (Q4_K_M GGUF) in a local `llama-server`
   ([llm_server.py](llm_server.py)) reads all lines and all 10 questions in
   one prompt and replies `Qn: [cited lines] yes|no`. The reply format is forced
   with a grammar, so it can never come back malformed ([answering.py](answering.py)).
4. **Evidence span**: the cited lines, trimmed at pauses to the part that
   matches the question, padded -0.10 s / +0.05 s.

Safety nets: speech-to-text stops at 33 s into a request (answers from a
partial transcript rather than timing out), a keyword fallback if the LLM
fails, and a guess if everything fails. It never returns an error.

No cloud APIs anywhere at inference time. Everything runs on the machine.

## Setup (one time)

From the repo root. Pick the backend for your GPU:

```powershell
# Windows
.\scripts\setup-medical.ps1 -Backend cuda      # NVIDIA
.\scripts\fetch-data.ps1 -SkipDrone            # training audio
```

```bash
# Linux x64
./scripts/setup-medical.sh cuda                # NVIDIA (CUDA 12.8 build)
# training audio: copy medical-appointment/data/audio from the organisers' repo
```

`vulkan` (AMD/Intel GPUs) and `cpu` also work. The script creates
`medical-appointment/.venv` (Python 3.12), installs packages, and downloads
llama.cpp release `b11029` plus the Gemma model (~5 GB) into `models/`, which
is gitignored.

## Run

```powershell
cd medical-appointment
$env:ASR_PROVIDER = "cuda"          # NVIDIA only: Parakeet on the GPU too
.\.venv\Scripts\python.exe api.py   # loads + warms up both models, serves :9054
```

Second terminal, same folder:

```powershell
.\.venv\Scripts\python.exe local_evaluator.py    # scores the 39 training conversations
```

Reference on the laptop: score **0.763**, 20.7 s per conversation on average,
32 s worst (the limit is 60 s). On a real GPU it should be much faster. Check
that the score matches ~0.76; if it's far off, something's misconfigured.

## Settings (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `ASR_PROVIDER` | `cpu` | `cuda` to run Parakeet on an NVIDIA GPU (needs the `cuda` setup) |
| `ASR_THREADS` | `6` | CPU threads for Parakeet (≈ number of performance cores) |
| `LLAMA_SERVER` | auto | path to `llama-server`; default: first found in `models/llama-cpp-cuda`, `-vulkan`, then `llama-cpp` |
| `LLAMA_MODEL` | `models/gemma-4-E4B-it-Q4_K_M.gguf` | the answering model |
| `LLAMA_NGL` | 99 (0 for the CPU build) | layers on the GPU |

The time limits are constants at the top of [example.py](example.py)
(`ASR_DEADLINE_S`, `REQUEST_BUDGET_S`).

## Going live

The evaluator needs a public URL. Easiest, from the machine running `api.py`:

```powershell
winget install --id Cloudflare.cloudflared
cloudflared tunnel --url http://localhost:9054
```

Submit `https://<printed-name>.trycloudflare.com/predict` (with `/predict`).
The address changes every time the tunnel restarts: never restart it during a run.

**Validation is unlimited; evaluation is ONE attempt.** Run a validation first,
check `outputs/api.log` (every line should say `via llm`, no `deadline hit`),
then evaluate.

## What we tried that did not help

So you don't repeat it (measured on the 39 training conversations, all
switchable in the code):

- Qwen3.5-4B / Qwen3.5-9B instead of Gemma: same accuracy, worse spans, slower.
- Prompt variants: "cite the whole exchange", questions before transcript,
  worked examples: no net gain.
- Splitting lines at commas or at pauses, a second focused pass per question
  ([tune_refine.py](tune_refine.py)): no net gain.
- Whisper small.en instead of Parakeet: similar text, 2x slower.

The remaining gap is choosing *which* lines the annotators marked (our spans
reach 0.61 tIoU; the best possible with our lines is ~0.82). Useful finding:
the annotated spans always start and end on pauses in the audio (it's stitched
TTS clips). A bigger model on a GPU (e.g. 27-31B) is the untested idea left.
