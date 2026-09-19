"""Kaggle GPU job: rebuild the dataset and train the detectors.

Runs as a Kaggle "script" kernel with GPU and internet on. Code comes from our
public repo, the 4K frames from the organisers' public repo, and the recorded
validation views from our private Kaggle dataset (mounted under /kaggle/input).
Only weights and training curves are kept as output (/kaggle/working).

Launched from the laptop with:   kaggle kernels push -p drone-flyby/kaggle
"""

import shutil
import subprocess
from pathlib import Path

OUR_REPO = 'https://github.com/paolotot/Nordic-AI-Cup-2026.git'
ORGANISERS = 'https://github.com/amboltio/Nordic-AI-Cup-2026.git'
RECORDINGS = Path('/kaggle/input/drone-flyby-recordings')
WORK = Path('/tmp/work')
OUTPUT = Path('/kaggle/working')
# (model, epochs, batch). Both T4s are used together; early stopping may end sooner.
RUNS = [('yolo11n', 50, 32), ('yolo11s', 40, 32)]


def sh(command, cwd=None):
    print(f'+ {command}', flush=True)
    subprocess.run(command, shell=True, check=True, cwd=cwd)


def copy_recordings(target):
    """Copy every recorded view (png + json sidecars), keeping sequence folders."""
    count = 0
    for png in RECORDINGS.rglob('*.png'):
        folder = target / png.parent.name
        folder.mkdir(parents=True, exist_ok=True)
        for sibling in png.parent.glob(png.stem + '*'):
            shutil.copy(sibling, folder / sibling.name)
        count += 1
    # The held-out labelled views must stay out of training there too.
    for holdout in RECORDINGS.rglob('holdout.json'):
        shutil.copy(holdout, target / 'holdout.json')
    print(f'copied {count} recorded views', flush=True)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    sh(f'git clone --depth 1 {OUR_REPO} repo', cwd=WORK)
    sh(f'git clone --depth 1 --filter=blob:none --sparse {ORGANISERS} organisers', cwd=WORK)
    sh('git sparse-checkout set drone-flyby/src/helsinki/images', cwd=WORK / 'organisers')
    drone = WORK / 'repo' / 'drone-flyby'
    shutil.copytree(WORK / 'organisers' / 'drone-flyby' / 'src' / 'helsinki' / 'images',
                    drone / 'src' / 'helsinki' / 'images', dirs_exist_ok=True)
    if RECORDINGS.exists():
        copy_recordings(drone / 'data' / 'recordings' / 'validation')

    sh('pip install -q ultralytics "faster-coco-eval>=1.7.2,<2"')
    sh('nvidia-smi --query-gpu=name,memory.total --format=csv')
    sh('python tools/make_cutouts.py', cwd=drone)
    sh('python tools/make_dataset.py', cwd=drone)

    for model, epochs, batch in RUNS:
        base = f'python tools/train.py --model {model} --epochs {epochs}'
        try:
            sh(f'{base} --device 0,1 --batch {batch}', cwd=drone)
        except subprocess.CalledProcessError:
            print(f'{model}: two-GPU run failed, retrying on one GPU', flush=True)
            try:
                sh(f'{base} --device 0 --batch {batch // 2}', cwd=drone)
            except subprocess.CalledProcessError as exc:
                print(f'{model} failed: {exc}', flush=True)
        run = drone / 'runs' / f'{model}_960'
        if run.exists():
            keep = OUTPUT / run.name
            keep.mkdir(parents=True, exist_ok=True)
            for name in ('weights/best.pt', 'results.csv', 'results.png',
                         'confusion_matrix_normalized.png', 'args.yaml'):
                if (run / name).exists():
                    shutil.copy(run / name, keep / Path(name).name)


if __name__ == '__main__':
    main()
