"""Time untrained YOLO detectors on this machine at the 960x540 view size.

Accuracy is meaningless here (COCO weights); this only answers "how big a
model fits in the 333 ms frame budget". Exports each model to OpenVINO once
(into models/) and times PyTorch and OpenVINO on the CPU.

The Iris Xe iGPU is left out on purpose: OpenVINO's GPU plugin hung for
minutes compiling even yolo11n, while OpenVINO on the CPU is already fast.

    python tools/speed_test.py
"""

import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'models'
IMGSZ = (544, 960)   # 540 rounded up to a multiple of 32
RUNS = 20


def median_ms(run):
    for _ in range(3):
        run()           # warm-up
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def main():
    import openvino as ov
    from ultralytics import YOLO

    MODELS.mkdir(exist_ok=True)
    image = np.random.randint(0, 255, (540, 960, 3), np.uint8)
    tensor = np.random.rand(1, 3, *IMGSZ).astype(np.float32)
    core = ov.Core()

    for name in ('yolo11n', 'yolo11s', 'yolo11m'):
        weights = MODELS / f'{name}.pt'
        if not weights.exists():
            YOLO(f'{name}.pt')                  # downloads into the working dir
            Path(f'{name}.pt').replace(weights)
        model = YOLO(str(weights))
        torch_ms = median_ms(lambda: model.predict(image, imgsz=IMGSZ, verbose=False,
                                                   device='cpu'))

        exported = MODELS / f'{name}_openvino_model'
        if not exported.exists():
            model.export(format='openvino', imgsz=IMGSZ)   # lands next to the weights
        compiled = core.compile_model(str(exported / f'{name}.xml'), 'CPU',
                                      {'PERFORMANCE_HINT': 'LATENCY'})
        request = compiled.create_infer_request()
        openvino_ms = median_ms(lambda: request.infer({0: tensor}))
        print(f'{name:8s} torch-cpu {torch_ms:5.0f} ms | openvino-cpu {openvino_ms:5.0f} ms',
              flush=True)


if __name__ == '__main__':
    main()
