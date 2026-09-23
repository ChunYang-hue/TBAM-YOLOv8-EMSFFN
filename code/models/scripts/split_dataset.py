"""
Create 8:1:1 train/val/test splits from original images.
Enhanced images derived from the same original image must stay in the same subset.
Five-fold cross-validation can be generated within the training set.
"""
import argparse
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-images', type=str, required=True)
    parser.add_argument('--out-splits', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    image_dir = Path(args.raw_images)
    out_dir = Path(args.out_splits)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in image_dir.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    random.shuffle(images)

    n = len(images)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train = images[:n_train]
    val = images[n_train:n_train + n_val]
    test = images[n_train + n_val:]

    for name, items in [('train', train), ('val', val), ('test', test)]:
        with open(out_dir / f'{name}.txt', 'w', encoding='utf-8') as f:
            for p in items:
                f.write(str(p.resolve()) + '\n')

    print(f'Total: {n}, train: {len(train)}, val: {len(val)}, test: {len(test)}')


if __name__ == '__main__':
    main()
