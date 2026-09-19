"""Kaggle GPU job: rebuild the dataset from GitHub and train the detectors.

Runs as a Kaggle "script" kernel with GPU and internet on. Nothing is uploaded
to Kaggle: the code comes from our public repo, the 4K frames from the
organisers' public repo. Only weights and training curves are kept as output
(/kaggle/working), the ~1 GB dataset lives in /tmp and is thrown away.

Launched from the laptop with:   kaggle kernels push -p drone-flyby/kaggle
"""

import shutil
import subprocess
from pathlib import Path

OUR_REPO = 'https://github.com/paolotot/Nordic-AI-Cup-2026.git'
ORGANISERS = 'https://github.com/amboltio/Nordic-AI-Cup-2026.git'
WORK = Path('/tmp/work')
OUTPUT = Path('/kaggle/working')
# (model, epochs). Early stopping (patience 15) usually ends runs sooner.
RUNS = [('yolo11n', 60), ('yolo11s', 60)]


def sh(command, cwd=None):
    print(f'+ {command}', flush=True)
    subprocess.run(command, shell=True, check=True, cwd=cwd)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    sh(f'git clone --depth 1 {OUR_REPO} repo', cwd=WORK)
    sh(f'git clone --depth 1 --filter=blob:none --sparse {ORGANISERS} organisers', cwd=WORK)
    sh('git sparse-checkout set drone-flyby/src/helsinki/images', cwd=WORK / 'organisers')
    drone = WORK / 'repo' / 'drone-flyby'
    shutil.copytree(WORK / 'organisers' / 'drone-flyby' / 'src' / 'helsinki' / 'images',
                    drone / 'src' / 'helsinki' / 'images', dirs_exist_ok=True)

    sh('pip install -q ultralytics "faster-coco-eval>=1.7.2,<2"')
    sh('nvidia-smi --query-gpu=name,memory.total --format=csv')
    sh('python tools/make_cutouts.py', cwd=drone)
    sh('python tools/make_dataset.py', cwd=drone)

    for model, epochs in RUNS:
        try:
            sh(f'python tools/train.py --model {model} --epochs {epochs} --device 0', cwd=drone)
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
