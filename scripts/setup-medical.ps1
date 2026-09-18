# One-time setup for medical-appointment on Windows: Python venv + packages,
# llama.cpp (llama-server) and the Gemma 4 E4B model. Everything it downloads
# is gitignored. Safe to re-run: existing downloads are skipped.
#
# From the repo root:
#   .\scripts\setup-medical.ps1 -Backend cuda     # NVIDIA GPU
#   .\scripts\setup-medical.ps1 -Backend vulkan   # any other GPU (Intel iGPU, AMD)
#   .\scripts\setup-medical.ps1 -Backend cpu      # no GPU
#
# Then also run .\scripts\fetch-data.ps1 -SkipDrone for the training audio
# (needed by local_evaluator.py and the server's warm-up).

param([ValidateSet('cuda', 'vulkan', 'cpu')][string]$Backend = 'cuda')

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$app = Join-Path $root 'medical-appointment'
$models = Join-Path $app 'models'
New-Item -ItemType Directory -Force -Path $models | Out-Null

# Pinned: the version everything was tested with.
$release = 'b11029'
$base = "https://github.com/ggml-org/llama.cpp/releases/download/$release"
$gguf = 'gemma-4-E4B-it-Q4_K_M.gguf'
$ggufUrl = "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/$gguf"

# --- Python ---------------------------------------------------------------
$venv = Join-Path $app '.venv'
if (-not (Test-Path $venv)) {
    Write-Host 'Creating venv (Python 3.12)...'
    py -3.12 -m venv $venv
}
$py = Join-Path $venv 'Scripts\python.exe'
$req = if ($Backend -eq 'cuda') { 'requirements-gpu.txt' } else { 'requirements.txt' }
if ($Backend -eq 'cuda') { & $py -m pip uninstall -y onnxruntime 2>$null | Out-Null }
Write-Host "Installing $req ..."
& $py -m pip install -q -r (Join-Path $app $req)

# --- llama.cpp ------------------------------------------------------------
$dir = Join-Path $models "llama-cpp-$Backend"
if ($Backend -eq 'cpu') { $dir = Join-Path $models 'llama-cpp' }
$zips = switch ($Backend) {
    'cuda'   { @("llama-$release-bin-win-cuda-12.4-x64.zip", 'cudart-llama-bin-win-cuda-12.4-x64.zip') }
    'vulkan' { @("llama-$release-bin-win-vulkan-x64.zip") }
    'cpu'    { @("llama-$release-bin-win-cpu-x64.zip") }
}
if (-not (Test-Path (Join-Path $dir 'llama-server.exe'))) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    foreach ($z in $zips) {
        Write-Host "Downloading $z ..."
        $tmp = Join-Path $models $z
        Invoke-WebRequest "$base/$z" -OutFile $tmp
        Expand-Archive $tmp -DestinationPath $dir -Force
        Remove-Item $tmp
    }
} else { Write-Host "llama.cpp already in $dir" }

# --- Model ----------------------------------------------------------------
$modelPath = Join-Path $models $gguf
if (-not (Test-Path $modelPath)) {
    Write-Host "Downloading $gguf (~5 GB) ..."
    Invoke-WebRequest $ggufUrl -OutFile $modelPath
} else { Write-Host "$gguf already present" }

Write-Host ''
Write-Host 'Done. To run:'
Write-Host '  cd medical-appointment'
if ($Backend -eq 'cuda') { Write-Host '  $env:ASR_PROVIDER = "cuda"      # Parakeet on the GPU too' }
Write-Host '  .\.venv\Scripts\python.exe api.py'
Write-Host '  # second terminal: .\.venv\Scripts\python.exe local_evaluator.py'
