"""
Check YOLO dataset integrity:
1. Each image has a corresponding label file.
2. Class IDs are within 0..nc-1.
3. Coordinates are normalized to [0, 1].
4. No empty label files.
5. No duplicate image paths in split files.
"""
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True)
    parser.add_argument('--labels', type=str, required=True)
    parser.add_argument('--nc', type=int, default=4)
    parser.add_argument('--splits', type=str, default=None)
    return parser.parse_args()


def check_images_labels(images_dir, labels_dir, nc):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

    errors = []
    images = [p for p in images_dir.rglob('*') if p.suffix.lower() in image_exts]

    for img in images:
        rel = img.relative_to(images_dir)
        label = labels_dir / rel.with_suffix('.txt')
        if not label.exists():
            errors.append(f'Missing label: {label}')
            continue

        lines = label.read_text(encoding='utf-8').strip().splitlines()
        if not lines:
            errors.append(f'Empty label: {label}')
            continue

        for line_no, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                errors.append(f'Bad format: {label}:{line_no}')
                continue
            cls, x, y, w, h = parts
            try:
                cls = int(cls)
                x, y, w, h = map(float, (x, y, w, h))
            except ValueError:
                errors.append(f'Non-numeric: {label}:{line_no}')
                continue

            if not (0 <= cls < nc):
                errors.append(f'Bad class id: {label}:{line_no} -> {cls}')
            for name, val in [('x', x), ('y', y), ('w', w), ('h', h)]:
                if not (0.0 <= val <= 1.0):
                    errors.append(f'Bad coordinate {name}: {label}:{line_no} -> {val}')

    return errors


def check_splits(splits_dir):
    splits_dir = Path(splits_dir)
    errors = []
    for split_file in splits_dir.glob('*.txt'):
        lines = [l.strip() for l in split_file.read_text(encoding='utf-8').splitlines() if l.strip()]
        if len(lines) != len(set(lines)):
            errors.append(f'Duplicate entries in {split_file}')
    return errors


def main():
    args = parse_args()
    errors = []
    errors.extend(check_images_labels(args.images, args.labels, args.nc))
    if args.splits:
        errors.extend(check_splits(args.splits))

    if errors:
        print(f'Found {len(errors)} errors:')
        for e in errors[:100]:
            print(e)
    else:
        print('Dataset check passed.')


if __name__ == '__main__':
    main()
