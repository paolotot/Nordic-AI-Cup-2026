"""Train a YOLO detector on data/synth. Same script on CPU and on a CUDA GPU.

    python tools/train.py                              # yolo11n, auto device
    python tools/train.py --model yolo11s --epochs 80  # bigger, on a GPU
    python tools/train.py --fraction 0.1 --epochs 1    # timing check

Weights land in runs/<name>/weights/best.pt. Rotation is already baked into
the pasted cutouts, so YOLO's own rotation stays off; vertical and horizontal
flips are free because the camera looks straight down.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', default='yolo11n')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--batch', type=int, default=None, help='default: 8 on CPU, 16 on GPU')
    parser.add_argument('--fraction', type=float, default=1.0, help='share of train images used')
    parser.add_argument('--device', default=None, help="'cpu', '0', ...; default: GPU if present")
    parser.add_argument('--name', default=None)
    arguments = parser.parse_args()

    import torch
    from ultralytics import YOLO

    device = arguments.device or ('0' if torch.cuda.is_available() else 'cpu')
    batch = arguments.batch or (8 if device == 'cpu' else 16)
    weights = ROOT / 'models' / f'{arguments.model}.pt'
    model = YOLO(str(weights) if weights.exists() else f'{arguments.model}.pt')
    model.train(
        data=str(ROOT / 'data' / 'synth' / 'data.yaml'),
        epochs=arguments.epochs,
        imgsz=arguments.imgsz,
        batch=batch,
        device=device,
        workers=4,
        fraction=arguments.fraction,
        project=str(ROOT / 'runs'),
        name=arguments.name or f'{arguments.model}_{arguments.imgsz}',
        exist_ok=True,
        degrees=0.0,
        flipud=0.5,
        fliplr=0.5,
        scale=0.3,
        mosaic=1.0,
        close_mosaic=5,
        patience=15,
        plots=True,
    )


if __name__ == '__main__':
    main()
