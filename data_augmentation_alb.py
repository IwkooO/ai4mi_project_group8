#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil
import cv2
import albumentations as A
import numpy as np

# -------------------------------------------------------
# Augmentation Strategies
# -------------------------------------------------------
GEOMETRIC_AUGS = [
    "hflip",
    "vflip", 
    "rotate",
    "shift_scale_rotate",
    "elastic",
    "grid_distort",
]

PIXEL_AUGS = [
    "brightness_contrast",
    "gamma",
    "gauss_noise",
    "blur",
    "motion_blur",
    "clahe",
    "coarse_dropout",
]

# -------------------------------------------------------
# Albumentations definitions
# -------------------------------------------------------
AUGMENTATIONS = {
    # Geometric
    "hflip": A.HorizontalFlip(p=1),
    "vflip": A.VerticalFlip(p=1),
    "rotate": A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=1),
    "shift_scale_rotate": A.ShiftScaleRotate(
        shift_limit=0.05, scale_limit=0.05, rotate_limit=10,
        border_mode=cv2.BORDER_REFLECT_101, p=1
    ),
    "elastic": A.ElasticTransform(
        alpha=30, sigma=4, alpha_affine=4,
        border_mode=cv2.BORDER_REFLECT_101, p=1
    ),
    "grid_distort": A.GridDistortion(
        num_steps=5, distort_limit=0.1,
        border_mode=cv2.BORDER_REFLECT_101, p=1
    ),
    
    # Pixel-level
    "brightness_contrast": A.RandomBrightnessContrast(
        brightness_limit=0.1, contrast_limit=0.1, p=1
    ),
    "gamma": A.RandomGamma(gamma_limit=(90, 110), p=1),
    "gauss_noise": A.GaussNoise(var_limit=(5.0, 15.0), mean=0, per_channel=False, p=1),
    "blur": A.Blur(blur_limit=3, p=1),
    "motion_blur": A.MotionBlur(blur_limit=3, p=1),
    "clahe": A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1),
    "coarse_dropout": A.CoarseDropout(
        max_holes=4, max_height=16, max_width=16,
        min_holes=1, min_height=8, min_width=8, fill_value=0, p=1
    ),
}

# -------------------------------------------------------
# Select augmentations based on strategy and multiplier
# -------------------------------------------------------
def select_augmentations(strategy, multiplier):
    """
    Select augmentations to achieve target multiplier.
    multiplier=8 means 8x data = original + 7 augmentations
    """
    num_augs_needed = multiplier - 1  # -1 because we keep original
    
    if num_augs_needed <= 0:
        return []
    
    if strategy == "geometric":
        augs = GEOMETRIC_AUGS[:num_augs_needed]
    elif strategy == "pixel":
        augs = PIXEL_AUGS[:num_augs_needed]
    elif strategy == "mixed":
        # Alternate between geometric and pixel for fairness
        all_augs = []
        geo_idx = 0
        pix_idx = 0
        for i in range(num_augs_needed):
            if i % 2 == 0 and geo_idx < len(GEOMETRIC_AUGS):
                all_augs.append(GEOMETRIC_AUGS[geo_idx])
                geo_idx += 1
            elif pix_idx < len(PIXEL_AUGS):
                all_augs.append(PIXEL_AUGS[pix_idx])
                pix_idx += 1
            elif geo_idx < len(GEOMETRIC_AUGS):
                all_augs.append(GEOMETRIC_AUGS[geo_idx])
                geo_idx += 1
        augs = all_augs
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return augs

# -------------------------------------------------------
# Apply augmentation
# -------------------------------------------------------
def apply_augmentation(image, mask, aug_name):
    aug = AUGMENTATIONS[aug_name]
    if aug_name in PIXEL_AUGS:
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

    img_count = 0
    aug_count = 0
    
    for img_file in (input_dir / "img").glob("*.png"):
        mask_file = input_dir / "gt" / img_file.name

        # Copy originals
        shutil.copy(img_file, output_dir / "img" / img_file.name)
        shutil.copy(mask_file, output_dir / "gt" / mask_file.name)
        img_count += 1

        # Load
        image = cv2.imread(str(img_file))
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)

        # Apply augmentations
        for aug in augmentations:
            if aug not in AUGMENTATIONS:
                print(f"Warning: Unknown augmentation '{aug}', skipping...")
                continue
            aug_img, aug_mask = apply_augmentation(image, mask, aug)

            aug_img_name = img_file.stem + f"_{aug}" + img_file.suffix
            aug_mask_name = mask_file.stem + f"_{aug}" + mask_file.suffix

            cv2.imwrite(str(output_dir / "img" / aug_img_name), aug_img)
            cv2.imwrite(str(output_dir / "gt" / aug_mask_name), aug_mask)
            aug_count += 1

    return img_count, aug_count

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Augment medical images with target multiplier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 8x data using mixed strategy
  python augment_with_multiplier.py --input_dir data/SEGTHOR --output_dir data/SEGTHOR_aug --multiplier 8 --strategy mixed
  
  # 5x data using only geometric transforms
  python augment_with_multiplier.py --input_dir data/SEGTHOR --output_dir data/SEGTHOR_aug --multiplier 5 --strategy geometric
  
  # 10x data using only pixel-level augmentations
  python augment_with_multiplier.py --input_dir data/SEGTHOR --output_dir data/SEGTHOR_aug --multiplier 10 --strategy pixel
        """
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Parent directory, e.g., data/SEGTHOR_CLEAN")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Parent output directory, e.g., data/SEGTHOR_CLEAN_aug")
    parser.add_argument("--multiplier", type=int, required=True,
                        help="Target dataset multiplier (e.g., 8 = 8x original size)")
    parser.add_argument("--strategy", type=str, default="mixed",
                        choices=["geometric", "pixel", "mixed"],
                        help="Augmentation strategy: geometric, pixel, or mixed")
    args = parser.parse_args()

    # Select augmentations
    augmentations = select_augmentations(args.strategy, args.multiplier)
    
    max_possible = len(GEOMETRIC_AUGS) if args.strategy == "geometric" else \
                   len(PIXEL_AUGS) if args.strategy == "pixel" else \
                   len(GEOMETRIC_AUGS) + len(PIXEL_AUGS)
    
    actual_multiplier = len(augmentations) + 1

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print(f"TARGET MULTIPLIER: {args.multiplier}x")
    print(f"STRATEGY: {args.strategy}")
    print(f"ACTUAL MULTIPLIER: {actual_multiplier}x (1 original + {len(augmentations)} augmentations)")
    if actual_multiplier < args.multiplier:
        print(f"⚠️  Note: Max possible for '{args.strategy}' strategy is {max_possible + 1}x")
    print("=" * 60)
    print(f"\nAugmentations selected ({len(augmentations)}):")
    for i, aug in enumerate(augmentations, 1):
        aug_type = "geometric" if aug in GEOMETRIC_AUGS else "pixel-level"
        print(f"  {i}. {aug:20s} [{aug_type}]")
    print()
    
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}\n")

    # Process train split with augmentation
    print("Processing training set...")
    orig_count, aug_count = process_split(input_dir / "train", output_dir / "train", augmentations)
    
    total_train = orig_count + aug_count
    print(f"  Original images: {orig_count}")
    print(f"  Augmented images: {aug_count}")
    print(f"  Total: {total_train} ({total_train / orig_count:.1f}x)")

    # Copy val split unchanged
    val_in = input_dir / "val"
    val_out = output_dir / "val"
    if val_in.exists():
        print(f"\nCopying validation set (unchanged)...")
        if val_out.exists():
            shutil.rmtree(val_out)
        shutil.copytree(val_in, val_out)
        val_count = len(list((val_out / "img").glob("*.png")))
        print(f"  Validation images: {val_count}")

    print("\n" + "=" * 60)
    print("✓ Data augmentation completed")
    print("=" * 60)

if __name__ == "__main__":
    main()