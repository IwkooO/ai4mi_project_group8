#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil
import cv2
import albumentations as A
import numpy as np

# -------------------------------------------------------
# Albumentations augmentations dictionary
# -------------------------------------------------------
AUGMENTATIONS = {
    "hflip": A.HorizontalFlip(p=1),
    "vflip": A.VerticalFlip(p=1),
    "rotate": A.Rotate(limit=15, p=1),
    "shift_scale_rotate": A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                                             rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=1),
    "random_crop": A.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), p=1),
    "elastic": A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1),
    "grid_distort": A.GridDistortion(p=1),
    "piecewise_affine": A.PiecewiseAffine(p=1),

    # pixel-level
    "brightness_contrast": A.RandomBrightnessContrast(p=1),
    "gamma": A.RandomGamma(p=1),
    "gauss_noise": A.GaussNoise(p=1),
    "blur": A.Blur(blur_limit=3, p=1),
    "motion_blur": A.MotionBlur(blur_limit=3, p=1),
    "clahe": A.CLAHE(p=1),

    # cutout-like
    "coarse_dropout": A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=1),

    # advanced
    # Note: MixUp & CutMix need pairs of images+labels; for segmentation we usually skip or handle differently.
}

# -------------------------------------------------------
# Apply augmentation
# -------------------------------------------------------
def apply_augmentation(image, mask, aug_name):
    aug = AUGMENTATIONS[aug_name]
    if aug_name in ["brightness_contrast", "gamma", "gauss_noise",
                    "blur", "motion_blur", "clahe", "coarse_dropout"]:
        # pixel-level: only apply to image
        augmented = aug(image=image)
        return augmented["image"], mask
    else:
        # geometric: apply to both
        augmented = aug(image=image, mask=mask)
        return augmented["image"], augmented["mask"]

# -------------------------------------------------------
# Process dataset
# -------------------------------------------------------
def process_split(input_dir, output_dir, augmentations):
    for subdir in ["img", "gt"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    for img_file in (input_dir / "img").glob("*.png"):
        mask_file = input_dir / "gt" / img_file.name

        # Copy originals
        shutil.copy(img_file, output_dir / "img" / img_file.name)
        shutil.copy(mask_file, output_dir / "gt" / mask_file.name)

        # Load
        image = cv2.imread(str(img_file))
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)

        # Apply augmentations
        for aug in augmentations:
            if aug not in AUGMENTATIONS:
                continue
            aug_img, aug_mask = apply_augmentation(image, mask, aug)

            aug_img_name = img_file.stem + f"_{aug}" + img_file.suffix
            aug_mask_name = mask_file.stem + f"_{aug}" + mask_file.suffix

            cv2.imwrite(str(output_dir / "img" / aug_img_name), aug_img)
            cv2.imwrite(str(output_dir / "gt" / aug_mask_name), aug_mask)

            # print(f"Applied {aug} -> {aug_img_name}, {aug_mask_name}")

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Parent directory, e.g., data/SEGTHOR_CLEAN")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Parent output directory, e.g., data/SEGTHOR_CLEAN_aug")
    parser.add_argument("--augmentations", type=str, required=True,
                        help="Comma-separated list: hflip,vflip,rotate,shift_scale_rotate,random_crop,"
                             "elastic,grid_distort,piecewise_affine,brightness_contrast,gamma,gauss_noise,"
                             "blur,motion_blur,clahe,coarse_dropout")
    args = parser.parse_args()

    augmentations = args.augmentations.split(",")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Process train split with augmentation
    process_split(input_dir / "train", output_dir / "train", augmentations)

    # Copy val split unchanged
    val_in = input_dir / "val"
    val_out = output_dir / "val"
    if val_in.exists():
        print(f"Copying validation set {val_in} -> {val_out}")
        if val_out.exists():
            shutil.rmtree(val_out)
        shutil.copytree(val_in, val_out)

    print("\n=== Data augmentation completed ===")

if __name__ == "__main__":
    main()
