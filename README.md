# Nordic AI Cup 2026 — team workspace

Working repo for the three challenges. Upstream template:
<https://github.com/amboltio/Nordic-AI-Cup-2026>

## The clock

| When | What |
|---|---|
| Sep 17–20, 2026 | Competition runs |
| **Sep 20, 16:00 CEST** | **Final submission closes** |
| Sep 20, 20:00 CEST | Top-5 teams hand in training code + weights for verification |

Submit at <https://cases.nordicaicup.com>. You give them a **URL and an API key** —
they call your server, you never upload a model.

## The rule that shapes everything

**One submission per challenge.** Unlimited validation runs, exactly one evaluation
run. So: iterate against the local evaluator and the validation endpoint, and only
fire the real one when you're done.

**No cloud AI APIs at inference time.** No OpenAI, no Azure Speech, no Gemini — the
model must run on your own machine. Using them *beforehand* (synthetic data,
distillation, labelling) is fine. Pretrained weights and outside datasets are fine.
The top-5 code handover is where cheating gets caught.

## Scoring

Per challenge, F1-style points by rank: **25, 18, 15, 12, 10, 8, 6, 4, 2, 1**.
Final standing = sum across the three. A solid finish in all three beats one win
and two blanks.

## The three challenges

| Dir | Port | Task | Metric |
|---|---|---|---|
| [medical-appointment/](medical-appointment/) | 9054 | Transcribe a doctor–patient MP3, answer 10 yes/no questions, timestamp the evidence | `0.4 × accuracy + 0.6 × mean tIoU` |
| [drone-flyby/](drone-flyby/) | 9053 | Detect 16 object classes in 4K aerial frames while steering a camera crop | COCO mAP @ IoU 0.50, macro-averaged |
| [survival-simulator/](survival-simulator/) | 9052 | Control a hivemind of herbivores for a 3000s sim | Survival time + fruit bonus, mean of 3 runs |

Each has its own upstream README with the full spec — read it before writing code.

## Repo layout

```
Nordic-AI-Cup-2026/
├── medical-appointment/   challenge code (api.py, example.py, dtos.py, local_evaluator.py)
├── drone-flyby/
├── survival-simulator/
├── docs/
│   └── azure-setup.md     how to host your endpoint publicly
├── scripts/
│   └── fetch-data.ps1     pulls the large data files from upstream
└── _upstream/             read-only clone of the organisers' repo (gitignored)
```

Large data (the MP3s, the 4K PNGs) is **gitignored** — a 450 MB git repo is misery
on venue wifi. Run `scripts/fetch-data.ps1` to pull it locally instead.

## First-time setup

```powershell
git clone https://github.com/paolotot/Nordic-AI-Cup-2026.git
cd Nordic-AI-Cup-2026
.\scripts\fetch-data.ps1          # downloads audio + drone frames from upstream
```

Then per challenge, its own venv (the dependency sets conflict):

```powershell
cd medical-appointment
py -3.12 -m venv .venv            # 3.12, NOT 3.14 — ML wheels don't exist for 3.14 yet
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running a challenge locally

```powershell
cd medical-appointment
.\.venv\Scripts\Activate.ps1
python api.py                     # serves 0.0.0.0:9054

# in a second terminal, same venv:
python local_evaluator.py         # scores your server on the 390 training questions
python local_evaluator.py --oracle    # what a perfect answer would score
python local_evaluator.py --verbose   # per-question breakdown
```

## Hardware reality check

This laptop: **i7-13700H, 14C/20T, 13.7 GB RAM, no NVIDIA GPU.**

Everything is CPU-bound. For medical-appointment that's the central constraint —
60 seconds per conversation to transcribe 2–4 minutes of audio, on CPU. Plan
around `faster-whisper` with `int8` quantisation and a small model, not `large-v3`.

See [docs/azure-setup.md](docs/azure-setup.md) for hosting — note that a student
Azure VM is likely *slower* than this laptop, so read that before assuming the
cloud helps.

## Working together

Branch per challenge, PR into `main`:

```powershell
git checkout -b medical/<what-youre-doing>
# ...work...
git push -u origin medical/<what-youre-doing>
```

Keep `main` deployable — it's what gets pulled onto the host machine.
