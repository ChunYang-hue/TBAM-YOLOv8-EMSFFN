"""
Apply geometric transformation, random rotation and noise to minority defect classes.
Only the training set should be augmented.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def random_rotate(image, angle_range=(-15, 15)):
    angle = random.uniform(*angle_range)
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def add_noise(image, sigma=10):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    out = image.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def geometric_transform(image):
    h, w = image.shape[:2]
    scale = random.uniform(0.9, 1.1)
    tx = random.uniform(-0.05, 0.05) * w
    ty = random.uniform(-0.05, 0.05) * h
    m = np.float32([[scale, 0, tx], [0, scale, ty]])
    return cv2.warpAffine(image, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--num-aug', type=int, default=1)
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in in_dir.rglob('*'):
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue
        image = cv2.imread(str(img_path))
        for i in range(args.num_aug):
            aug = geometric_transform(image)
            aug = random_rotate(aug)
            aug = add_noise(aug)
            out_path = out_dir / f'{img_path.stem}_aug{i}{img_path.suffix}'
            cv2.imwrite(str(out_path), aug)


if __name__ == '__main__':
    main()
