# Downloads the large competition data (audio, 4K drone frames) from the
# organisers' repo and copies it into place. That data is gitignored in our repo.
#
# Usage, from the repo root:   .\scripts\fetch-data.ps1
# Skip the ~450 MB drone frames: .\scripts\fetch-data.ps1 -SkipDrone

param([switch]$SkipDrone)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$upstream = Join-Path $root '_upstream'
$url = 'https://github.com/amboltio/Nordic-AI-Cup-2026.git'

if (-not (Test-Path $upstream)) {
    Write-Host "Cloning organisers' repo (metadata only)..."
    git clone --depth 1 --filter=blob:none --sparse $url $upstream
} else {
    Write-Host 'Updating organisers repo...'
    git -C $upstream pull --ff-only
}

$dirs = @('medical-appointment', 'survival-simulator')
if (-not $SkipDrone) { $dirs += @('drone-flyby', 'images') }

Write-Host "Downloading: $($dirs -join ', ')  (drone frames are large, be patient)"
git -C $upstream sparse-checkout set @dirs

# Copy only the gitignored data folders — never overwrite our own code.
$data = @('medical-appointment\data\audio')
if (-not $SkipDrone) { $data += @('drone-flyby\src\helsinki\images', 'drone-flyby\images', 'images') }

foreach ($d in $data) {
    $src = Join-Path $upstream $d
    $dst = Join-Path $root $d
    if (Test-Path $src) {
        New-Item -ItemType Directory -Force -Path $dst | Out-Null
        Copy-Item "$src\*" $dst -Recurse -Force
        Write-Host "  ok  $d"
    }
}
Write-Host 'Done.'
