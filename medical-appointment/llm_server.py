"""Start and stop the local llama.cpp server that hosts the answering LLM.

Defaults to the Vulkan build on the Intel iGPU (about 3x faster prompt
processing than the CPU build on this laptop). Override with env vars on
another machine:

    LLAMA_SERVER   path to llama-server(.exe)
    LLAMA_MODEL    path to the GGUF
    LLAMA_NGL      layers to offload to the GPU (0 for CPU-only)
"""

import atexit
import os
import subprocess
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
DEFAULT_SERVER = HERE / 'models' / 'llama-cpp-vulkan' / 'llama-server.exe'
DEFAULT_MODEL = HERE / 'models' / 'gemma-4-E4B-it-Q4_K_M.gguf'
PORT = 8080
URL = f'http://127.0.0.1:{PORT}'


def start(model=None, server=None, ngl=None, threads=0, ctx=8192,
          log_name=None) -> subprocess.Popen:
    model = Path(model or os.environ.get('LLAMA_MODEL', DEFAULT_MODEL))
    server = Path(server or os.environ.get('LLAMA_SERVER', DEFAULT_SERVER))
    ngl = int(ngl if ngl is not None else os.environ.get('LLAMA_NGL', 99))

    (HERE / 'outputs').mkdir(exist_ok=True)
    log = open(HERE / 'outputs' / (log_name or f'llama-server-{model.stem}.log'), 'w')
    cmd = [str(server), '-m', str(model), '--port', str(PORT), '-c', str(ctx),
           '--no-webui', '-np', '1', '-ngl', str(ngl)]
    if ngl > 0:
        # Fully offloaded: the CPU threads only feed the iGPU. Few threads and
        # no spin-polling, or they steal cores from the next request's ASR
        # (measured: ASR 33.3 s -> 29.7 s after an answer, answer time same).
        cmd += ['-t', str(threads or 4), '--poll', '0']
    elif threads:
        cmd += ['-t', str(threads)]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    atexit.register(proc.terminate)

    for _ in range(300):
        try:
            if requests.get(f'{URL}/health', timeout=1).ok:
                return proc
        except requests.RequestException:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f'llama-server exited, see {log.name}')
        time.sleep(1)
    raise RuntimeError('llama-server did not come up')
