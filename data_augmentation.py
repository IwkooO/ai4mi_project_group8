#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np

def flip(img):
    return cv2.flip(img, 1)  # horizontal flip

def rotate(img, angle=15):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST)

def add_noise(img):
    noise = np.random.normal(0, 10, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

AUGMENT_FUNCS = {
    "flip": flip,
    "rotate": lambda img: rotate(img, angle=15),
    "rotate_neg": lambda img: rotate(img, angle=-15),
    "noise": add_noise,
}

def process_dir(input_dir, output_dir, augmentations):
    for subdir in ["img", "gt"]:
        in_path = Path(input_dir) / subdir
        out_path = Path(output_dir) / subdir
        out_path.mkdir(parents=True, exist_ok=True)

        for file in in_path.glob("*.png"):
            # Copy original
            shutil.copy(file, out_path / file.name)

            # Apply augmentations
            img = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
            for aug in augmentations:
                if aug not in AUGMENT_FUNCS:
                    continue
                aug_img = AUGMENT_FUNCS[aug](img)
                aug_name = file.stem + f"_{aug}" + file.suffix
                cv2.imwrite(str(out_path / aug_name), aug_img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmentations", type=str, required=True,
                        help="Comma-separated list: flip,rotate,rotate_neg,noise")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Parent directory, e.g., data/SEGTHOR_CLEAN")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Parent output directory, e.g., data/SEGTHOR_CLEAN_aug")
    args = parser.parse_args()

    augmentations = args.augmentations.split(",")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # 1. Process only train (with augmentation)
    train_dir = input_dir / "train"
    out_train_dir = output_dir / "train"
    process_dir(train_dir, out_train_dir, augmentations)

    # 2. Copy val unchanged
    val_dir = input_dir / "val"
    out_val_dir = output_dir / "val"
    if val_dir.exists():
        print(f"Copying validation set from {val_dir} -> {out_val_dir}")
        if out_val_dir.exists():
            shutil.rmtree(out_val_dir)
        shutil.copytree(val_dir, out_val_dir)

if __name__ == "__main__":
    main()
