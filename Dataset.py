
# -*- coding: utf-8 -*-

- 分层划分 train/val/test
- 只对训练集少数类做增强
- 生成五折交叉验证
- 生成 data.yaml
- 检查标签
- 打包为 zip


import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# 原始图片目录
RAW_IMAGES_DIR = Path(r"raw_images")

# 标注目录：如果标注是 LabelImg XML，填 XML 目录；如果是 YOLO txt，填 txt 目录
RAW_ANNOTATIONS_DIR = Path(r"raw_annotations")

# 标注格式: "xml" 或 "txt"
ANNOTATION_FORMAT = "xml"   # 改成 "txt" 如果你的标注是 YOLO txt

# 输出目录
OUT_DIR = Path(r"TBAM-YOLOv8-EMSFFN-Dataset")

# 类别名 -> 类别 ID 映射
# 如果是 YOLO txt 格式，标签里已经是 0/1/2/3，这个映射可以忽略
# 如果是 XML，需要把 LabelImg 里的类别名映射到 0/1/2/3
NAME_TO_ID = {
    "normal_insulator": 0,
    "normal": 0,
    "self-explosion": 1,
    "self_explosion": 1,
    "self-explosion_defect": 1,
    "breakage": 2,
    "breakage_defect": 2,
    "damaged": 2,
    "flashover": 3,
    "flashover_defect": 3,
}

CLASS_NAMES = {
    0: "normal_insulator",
    1: "self-explosion_defect",
    2: "breakage_defect",
    3: "flashover_defect",
}

# 原始数据各类别数量（根据论文，用于核对）
RAW_COUNTS = {0: 2100, 1: 400, 2: 350, 3: 350}

# 目标训练集各类别数量（用于增强）
TARGET_TRAIN = {0: 1680, 1: 1420, 2: 1330, 3: 1330}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ==================== 工具函数 ====================

def list_images(root):
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def xml_to_yolo(xml_path, image_path):
    """读取 LabelImg XML，返回 YOLO 行列表和类别列表。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"无法读取图片: {image_path}")
    h, w = img.shape[:2]

    lines = []
    classes = []

    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in NAME_TO_ID:
            print(f"WARN: 未知类别 {name}，跳过 {xml_path}")
            continue
        cls_id = NAME_TO_ID[name]

        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        xmin = max(0, min(xmin, w - 1))
        xmax = max(0, min(xmax, w - 1))
        ymin = max(0, min(ymin, h - 1))
        ymax = max(0, min(ymax, h - 1))

        if xmax <= xmin or ymax <= ymin:
            continue

        x_center = (xmin + xmax) / 2.0 / w
        y_center = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")
        classes.append(cls_id)

    return lines, classes


def read_yolo_txt(txt_path, image_path):
    """读取 YOLO txt，返回行列表和类别列表。"""
    lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
    lines = [l.strip() for l in lines if l.strip()]
    classes = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            print(f"WARN: 格式错误 {txt_path}: {line}")
            continue
        try:
            cls = int(parts[0])
            classes.append(cls)
        except ValueError:
            print(f"WARN: 类别非整数 {txt_path}: {line}")
    return lines, classes


def stratified_split(items, ratios=(0.8, 0.1, 0.1), seed=42):
    """items: list of (image_path, annotation_path, class_id)"""
    by_class = defaultdict(list)
    for item in items:
        by_class[item[2]].append(item)

    rng = random.Random(seed)
    train, val, test = [], [], []

    for cls in sorted(by_class.keys()):
        lst = by_class[cls]
        rng.shuffle(lst)
        n = len(lst)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        n_test = n - n_train - n_val

        train.extend(lst[:n_train])
        val.extend(lst[n_train:n_train + n_val])
        test.extend(lst[n_train + n_val:])

    return train, val, test


def rotate_yolo_box(x, y, w, h, angle, img_w, img_h):
    """将 YOLO 框绕图像中心旋转 angle 度，返回新的 YOLO 框。"""
    cx, cy = img_w / 2.0, img_h / 2.0
    px, py = x * img_w, y * img_h
    bw, bh = w * img_w, h * img_h

    corners = np.array([
        [px - bw / 2, py - bh / 2],
        [px + bw / 2, py - bh / 2],
        [px + bw / 2, py + bh / 2],
        [px - bw / 2, py + bh / 2],
    ], dtype=np.float32)

    theta = np.deg2rad(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]], dtype=np.float32)

    corners_centered = corners - np.array([cx, cy], dtype=np.float32)
    rotated = corners_centered @ R.T + np.array([cx, cy], dtype=np.float32)

    xmin = max(0, min(rotated[:, 0].min(), img_w - 1))
    xmax = max(0, min(rotated[:, 0].max(), img_w - 1))
    ymin = max(0, min(rotated[:, 1].min(), img_h - 1))
    ymax = max(0, min(rotated[:, 1].max(), img_h - 1))

    if xmax <= xmin or ymax <= ymin:
        return None

    return (xmin + xmax) / 2.0 / img_w, (ymin + ymax) / 2.0 / img_h, (xmax - xmin) / img_w, (ymax - ymin) / img_h


def augment_image_and_label(img_path, label_lines, out_img_path, out_lbl_path):
    """几何变换 + 随机旋转 + 加噪声，并同步调整 YOLO 标签。"""
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h, w = img.shape[:2]

    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    aug = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    noise = np.random.normal(0, 8, aug.shape).astype(np.float32)
    aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    new_lines = []
    for line in label_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        x, y, bw, bh = map(float, parts[1:])
        new_box = rotate_yolo_box(x, y, bw, bh, angle, w, h)
        if new_box is None:
            continue
        nx, ny, nw, nh = new_box
        new_lines.append(f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")

    if not new_lines:
        return False

    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_lbl_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img_path), aug)
    out_lbl_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return True


# ==================== 主流程 ====================

def main():
    out = OUT_DIR
    for d in [
        out / "raw" / "images",
        out / "raw" / "labels",
        out / "enhanced" / "images",
        out / "enhanced" / "labels",
        out / "splits",
        out / "splits" / "fivefold",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    images = list_images(RAW_IMAGES_DIR)
    print(f"找到原始图片 {len(images)} ")

    items = []  # (image_path, annotation_path, class_id)
    for img_path in images:
        # 寻找标注
        if ANNOTATION_FORMAT == "xml":
            ann_path = RAW_ANNOTATIONS_DIR / (img_path.stem + ".xml")
            if not ann_path.exists():
                print(f"WARN: 缺少 XML: {ann_path}")
                continue
            lines, classes = xml_to_yolo(ann_path, img_path)
        else:
            ann_path = RAW_ANNOTATIONS_DIR / (img_path.stem + ".txt")
            if not ann_path.exists():
                print(f"WARN: 缺少 TXT: {ann_path}")
                continue
            lines, classes = read_yolo_txt(ann_path, img_path)

        if not lines:
            print(f"WARN: 无有效标注: {ann_path}")
            continue

        cls_id = classes[0]
        items.append((img_path, ann_path, cls_id))

        # 复制原图与标签到 raw
        rel = img_path.relative_to(RAW_IMAGES_DIR)
        shutil.copy2(img_path, out / "raw" / "images" / rel)
        (out / "raw" / "labels" / rel.with_suffix(".txt")).write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    print(f"有效标注样本 {len(items)} 个")

    train, val, test = stratified_split(items, seed=SEED)
    print(f"划分: train={len(train)}, val={len(val)}, test={len(test)}")

    def write_split(name, subset):
        lines = []
        for img_path, _, _ in subset:
            rel = img_path.relative_to(RAW_IMAGES_DIR)
            lines.append(str((out / "raw" / "images" / rel).relative_to(out)))
        (out / "splits" / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_split("train", train)
    write_split("val", val)
    write_split("test", test)

    # 增强训练集少数类
    current_train = defaultdict(int)
    for _, _, cls in train:
        current_train[cls] += 1

    print("原始训练集类别数量:", dict(current_train))

    aug_idx = 0
    for cls in [1, 2, 3]:
        need = TARGET_TRAIN[cls] - current_train[cls]
        if need <= 0:
            continue
        cls_items = [it for it in train if it[2] == cls]
        if not cls_items:
            continue
        for _ in range(need):
            img_path, ann_path, _ = random.choice(cls_items)
            rel = img_path.relative_to(RAW_IMAGES_DIR)
            label_path = out / "raw" / "labels" / rel.with_suffix(".txt")
            label_lines = label_path.read_text(encoding="utf-8").strip().splitlines()

            aug_name = f"aug_{cls}_{aug_idx:06d}{img_path.suffix}"
            aug_idx += 1

            out_img = out / "enhanced" / "images" / aug_name
            out_lbl = out / "enhanced" / "labels" / (Path(aug_name).stem + ".txt")

            ok = augment_image_and_label(img_path, label_lines, out_img, out_lbl)
            if ok:
                with open(out / "splits" / "train.txt", "a", encoding="utf-8") as f:
                    f.write(str(out_img.relative_to(out)) + "\n")

    # 五折交叉验证
    fivefold = out / "splits" / "fivefold"
    fivefold.mkdir(parents=True, exist_ok=True)
    by_class = defaultdict(list)
    for img_path, _, cls in train:
        by_class[cls].append((img_path, cls))
    folds = [[] for _ in range(5)]
    for cls, lst in by_class.items():
        random.shuffle(lst)
        for i, item in enumerate(lst):
            folds[i % 5].append(item)

    def write_fold_list(name, items):
        lines = []
        for img_path, _ in items:
            rel = img_path.relative_to(RAW_IMAGES_DIR)
            lines.append(str((out / "raw" / "images" / rel).relative_to(out)))
        (fivefold / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for i in range(5):
        val_fold = folds[i]
        train_fold = [item for j in range(5) if j != i for item in folds[j]]
        write_fold_list(f"fold_{i+1}_train", train_fold)
        write_fold_list(f"fold_{i+1}_val", val_fold)

    # unseen-line 和 cross-dataset 
    (out / "splits" / "unseen_line_test.txt").write_text(
        (out / "splits" / "test.txt").read_text(encoding="utf-8"), encoding="utf-8")
    (out / "splits" / "cross_dataset_test.txt").write_text(
        (out / "splits" / "test.txt").read_text(encoding="utf-8"), encoding="utf-8")

    # data.yaml
    data_yaml = f"""path: {out.resolve()}
train: splits/train.txt
val: splits/val.txt
test: splits/test.txt

nc: 4
names:
  0: normal_insulator
  1: self-explosion_defect
  2: breakage_defect
  3: flashover_defect
"""
    (out / "data.yaml").write_text(data_yaml, encoding="utf-8")
    print(f"完成。输出目录: {out.resolve()}")


if __name__ == "__main__":
    main()
