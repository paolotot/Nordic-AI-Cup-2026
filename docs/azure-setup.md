# Hosting your solution on Azure

## Why you need this at all

The evaluation server does not run your code. It sends HTTP POSTs to a URL **you**
give it. So something of yours has to be reachable from the public internet for the
whole duration of an attempt, on a fixed address.

Two ways to get that:

| Option | Good for | Bad for |
|---|---|---|
| **A. Azure VM** (what the organisers recommend) | Stable, runs while your laptop sleeps, you can pick a bigger CPU than your laptop has | Costs credit, setup time, student quota is small |
| **B. Your laptop + a tunnel** (`cloudflared`) | Zero setup, uses your i7-13700H, no quota fights | Laptop must stay awake, on wifi, and the venue wifi must behave |

Realistic advice: **build and validate locally (B), submit from whichever is faster
on the day.** Set up the Azure VM early so it's there if you want it, but don't
assume it will be faster than your laptop — see the quota note below.

---

## Option A: Azure VM, step by step

### 1. Get the subscription

Go to <https://azure.microsoft.com/free/students>. Sign in with your **university
email**. You get **$100 of credit for 12 months, no credit card required**. Approval
is normally instant, but if your university isn't auto-recognised it can take a day
or two — so do this *now*, not on the 20th.

### 2. Create the VM

Portal → **Virtual machines** → **Create** → *Azure virtual machine*.

| Field | Set it to | Why |
|---|---|---|
| Resource group | `nordic-ai-cup` (create new) | one-click cleanup afterwards |
| Region | **Sweden Central** or **North Europe** | lowest latency to the evaluator |
| Image | **Ubuntu Server 24.04 LTS** | everything below assumes this |
| Size | see the size note ↓ | |
| Authentication | **SSH public key**, new key pair | download the `.pem`, you get it once |
| Public inbound ports | allow **SSH (22)** only for now | the challenge ports come next |
| OS disk | 64 GB Premium SSD | model weights are a few GB |

Hit Create, download the private key when prompted, note the **public IP**.

> **Make the IP static.** VM → *Networking* → click the IP → *Static*. On a dynamic
> IP your address changes if the VM ever restarts, and your submitted URL dies
> mid-attempt.

### 3. Size note — read this before you pick a VM

Student subscriptions ship with a **low vCPU quota (often 4–10 total) and no GPU
quota at all**. GPU sizes (NC/ND-series) will almost certainly be refused. You can
request a quota increase under *Subscriptions → Usage + quotas → Request increase*,
but approval for GPU on a student account is unlikely.

So assume **CPU only**. Reasonable picks inside a small quota:

- `Standard_F8s_v2` — 8 compute-optimised vCPU, best CPU-per-core, good for Whisper
- `Standard_D8s_v5` — 8 general vCPU, 32 GB RAM
- `Standard_F4s_v2` — 4 vCPU, the fallback if quota is 4

**Your laptop has 14 cores / 20 threads.** An 8-vCPU VM is *slower* than your own
machine. That's the whole reason Option B is on the table.

### 4. Open the challenge ports

This is the step people forget, and it looks exactly like a broken model:
the evaluator just times out.

VM → **Networking** → **Network settings** → *Add inbound port rule*:

| Field | Value |
|---|---|
| Source | `Any` |
| Source port ranges | `*` |
| Destination | `Any` |
| Service | `Custom` |
| Destination port ranges | `9052,9053,9054` |
| Protocol | `TCP` |
| Action | `Allow` |
| Priority | `1010` |
| Name | `nordic-ai-cup-ports` |

Ubuntu's own firewall (`ufw`) is inactive by default on Azure images, so the NSG
rule is normally all you need. If you enabled `ufw`, also run
`sudo ufw allow 9052:9054/tcp`.

### 5. Connect and set up

From PowerShell on your laptop:

```powershell
# lock down the key file or ssh refuses it
icacls .\nordic-key.pem /inheritance:r
icacls .\nordic-key.pem /grant:r "$($env:USERNAME):(R)"

ssh -i .\nordic-key.pem azureuser@<YOUR_PUBLIC_IP>
```

Then on the VM:

```bash
sudo apt update && sudo apt install -y python3.12-venv python3-pip ffmpeg git tmux
git clone https://github.com/paolotot/Nordic-AI-Cup-2026.git
cd Nordic-AI-Cup-2026/medical-appointment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 6. Run it so it survives you closing the laptop

`tmux` keeps the process alive after you disconnect:

```bash
tmux new -s medical
source .venv/bin/activate
python api.py
# detach with: Ctrl+B then D
# come back with: tmux attach -t medical
```

### 7. Verify from the outside before you submit

From your **laptop**, not the VM — this is the only test that proves the evaluator
can reach you:

```powershell
curl http://<YOUR_PUBLIC_IP>:9054/
# expect: "Your endpoint is running!"
```

If that hangs: the NSG rule is wrong, or the server is bound to `127.0.0.1`
instead of `0.0.0.0`. `api.py` already uses `0.0.0.0` — leave it alone.

**Submit** `http://<YOUR_PUBLIC_IP>:9054/predict` — the full URL including the
path, because the organisers use the string exactly as you type it.

### 8. Stop the VM when you're not using it

VM → **Stop** (this is "deallocate" — billing stops). A running 8-vCPU VM burns
roughly $10–15/week of your $100. Just remember the IP is only preserved across
restarts if you made it **static** in step 2.

---

## Option B: laptop + Cloudflare tunnel

No account, no card, one command. Good for quick validation runs.

```powershell
winget install --id Cloudflare.cloudflared
# start your API first (python api.py), then:
cloudflared tunnel --url http://localhost:9054
```

It prints a public `https://something-random.trycloudflare.com` URL. Submit that
**plus the path**: `https://something-random.trycloudflare.com/predict`.

Caveats, and they are real:
- The URL changes every time you restart the tunnel. Never restart it mid-attempt.
- Free quick-tunnels are best-effort and can drop. Five consecutive timeouts ends
  your medical-appointment attempt permanently.
- Your laptop must not sleep. Set power mode to *Best performance*, plugged in,
  and disable sleep in Windows settings.

Use it for validation freely. For the one real submission, decide on the day based
on which host actually answers faster and more reliably.

---

## Rule check: is Azure even allowed?

Yes, with one distinction that matters:

- **Hosting your own model on an Azure VM: allowed and recommended.** That's your
  server, running your weights.
- **Calling a cloud AI API at inference time — Azure OpenAI, Speech-to-Text,
  OpenAI, Gemini: not allowed.** The model has to run on its own.
- **Using cloud APIs while training/preparing: allowed.** Generating synthetic
  training data or distilling a model beforehand is fine.

The top 5 teams hand over training code and weights for verification after the
deadline, so anything that phones home at inference will get caught there.
