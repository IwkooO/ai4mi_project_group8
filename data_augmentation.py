#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np
import random

# ------------------------------
# Augmentation functions
# ------------------------------

def hflip(img, p=0.5):
    if random.random() < p:
        return cv2.flip(img, 1)
    return img

def random_rotation(img, max_angle=15):
    angle = random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST)

def random_scale(img, scale_range=(0.9, 1.1)):
    h, w = img.shape[:2]
    scale = random.uniform(*scale_range)
    new_w, new_h = int(w * scale), int(h * scale)
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    if scale >= 1.0:
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        return scaled[start_y:start_y+h, start_x:start_x+w]
    else:
        out = np.zeros_like(img)
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        out[start_y:start_y+new_h, start_x:start_x+new_w] = scaled
        return out

def random_crop_resize(img, crop_size=0.8):
    h, w = img.shape[:2]
    ch, cw = int(h * crop_size), int(w * crop_size)
    start_x = random.randint(0, w - cw)
    start_y = random.randint(0, h - ch)
    crop = img[start_y:start_y+ch, start_x:start_x+cw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)

def elastic_deformation(img, alpha=10, sigma=5):
    random_state = np.random.RandomState(None)
    shape = img.shape[:2]

    dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1),
                          (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1),
                          (17, 17), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_REFLECT_101)

def intensity_shift(img, brightness=0.2, contrast=0.2):
    img = img.astype(np.float32)
    b = random.uniform(-brightness, brightness) * 255
    c = 1.0 + random.uniform(-contrast, contrast)
    out = img * c + b
    return np.clip(out, 0, 255).astype(np.uint8)

def gaussian_blur(img, p=0.2, sigma_range=(1, 2)):
    if random.random() < p:
        sigma = random.uniform(*sigma_range)
        ksize = int(2 * round(3 * sigma) + 1)
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return img

# ------------------------------
# Augmentation registry
# ------------------------------
AUGMENT_FUNCS = {
    "hflip": hflip,
    "rotate": random_rotation,
    "scale": random_scale,
    "crop": random_crop_resize,
    "elastic": elastic_deformation,
    "intensity": intensity_shift,
    "blur": gaussian_blur,
}

# ------------------------------
# Processing
# ------------------------------
def process_dir(input_dir, output_dir, augmentations):
    for subdir in ["img", "gt"]:
        in_path = Path(input_dir) / subdir
        out_path = Path(output_dir) / subdir
        out_path.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing directory: {in_path}")

        for file in in_path.glob("*.png"):
            # print(f"  Copying original: {file.name}")
            shutil.copy(file, out_path / file.name)

            img = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
            for aug in augmentations:
                if aug not in AUGMENT_FUNCS:
                    continue
                aug_img = AUGMENT_FUNCS[aug](img)
                aug_name = file.stem + f"_{aug}" + file.suffix
                cv2.imwrite(str(out_path / aug_name), aug_img)
                # print(f"    Applied augmentation: {aug} -> {aug_name}")

        print(f"Completed processing of {subdir} folder")

    print(f"\nAll augmentations completed for {input_dir} -> {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmentations", type=str, required=True,
                        help="Comma-separated list: hflip,rotate,scale,crop,elastic,intensity,blur")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Parent directory, e.g., data/SEGTHOR_CLEAN")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Parent output directory, e.g., data/SEGTHOR_CLEAN_aug")
    args = parser.parse_args()

    augmentations = args.augmentations.split(",")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    train_dir = input_dir / "train"
    out_train_dir = output_dir / "train"
    process_dir(train_dir, out_train_dir, augmentations)

    val_dir = input_dir / "val"
    out_val_dir = output_dir / "val"
    if val_dir.exists():
        print(f"\nCopying validation set from {val_dir} -> {out_val_dir}")
        if out_val_dir.exists():
            shutil.rmtree(out_val_dir)
        shutil.copytree(val_dir, out_val_dir)
        print("Validation set copied successfully")

    print("\n=== Data augmentation pipeline finished ===")

if __name__ == "__main__":
    main()
