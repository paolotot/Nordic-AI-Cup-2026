"""Import a makesense.ai YOLO export into the recordings as hand labels.

data/to_label/images/L<level>_<stem>.png were labelled with the classes in
data/to_label/classes.txt (the 16 object classes, then "unsure"). Each exported
.txt becomes <stem>.labels.json next to the recording, and "unsure" boxes become
<stem>.ignore.json (painted over in training). Images without an exported .txt
are left alone unless --empty-means-no-objects is given.

    python tools/import_labels.py data/to_label/labels_export.zip
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dtos import OBJECT_CLASSES  # noqa: E402

FOLDER = ROOT / 'data' / 'to_label'
RECORDINGS = ROOT / 'data' / 'recordings' / 'validation'
WIDTH, HEIGHT = 960, 540


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('export', help='the .zip (or a folder of .txt files)')
    parser.add_argument('--empty-means-no-objects', action='store_true',
                        help='treat images with no exported file as labelled empty')
    arguments = parser.parse_args()

    names = (FOLDER / 'classes.txt').read_text().split()
    source = Path(arguments.export)
    texts = {}
    if source.suffix == '.zip':
        with zipfile.ZipFile(source) as archive:
            for member in archive.namelist():
                if member.endswith('.txt') and Path(member).name != 'classes.txt':
                    texts[Path(member).stem] = archive.read(member).decode()
    else:
        texts = {p.stem: p.read_text() for p in source.glob('*.txt') if p.name != 'classes.txt'}

    images = sorted((FOLDER / 'images').glob('*.png'))
    # The images to label were taken from the largest recorded run; stems
    # repeat across runs, so only look there.
    run = max((p for p in RECORDINGS.iterdir() if p.is_dir()),
              key=lambda p: len(list(p.glob('*.png'))))
    counts, written = {}, 0
    for image in images:
        if image.stem not in texts and not arguments.empty_means_no_objects:
            continue
        stem = image.stem.split('_', 1)[1]                 # drop the L<level>_ prefix
        matches = list(run.glob(f'{stem}.png'))
        if len(matches) != 1:
            print(f'skip {image.name}: {len(matches)} recordings match')
            continue
        labels, ignore = [], []
        for line in texts.get(image.stem, '').splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            name = names[int(parts[0])]
            cx, cy, w, h = (float(v) for v in parts[1:])
            box = [round((cx - w / 2) * WIDTH, 1), round((cy - h / 2) * HEIGHT, 1),
                   round((cx + w / 2) * WIDTH, 1), round((cy + h / 2) * HEIGHT, 1)]
            if name == 'unsure':
                ignore.append(box)
            elif name in OBJECT_CLASSES:
                labels.append({'object_id': name, 'bbox': box})
                counts[name] = counts.get(name, 0) + 1
        base = matches[0].with_suffix('')
        Path(f'{base}.labels.json').write_text(json.dumps(labels))
        Path(f'{base}.ignore.json').write_text(json.dumps(ignore))
        written += 1
    print(f'{written} views labelled; objects per class: {counts}')


if __name__ == '__main__':
    main()
