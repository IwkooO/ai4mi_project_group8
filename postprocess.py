"""
Post-processing for 3D segmentation masks

Supports 3 modes:
  --mode lcc      → Keep only Largest Connected Component (LCC)
  --mode morph    → Apply morphological cleaning (opening + closing)
  --mode lcc+morph    → Apply LCC first, then morphological cleaning
"""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import (
    label,
    binary_opening,
    binary_closing,
    generate_binary_structure,
)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask"""
    labeled, num = label(mask)
    if num < 1:
        return mask
    # label of largest non-background component
    largest = np.argmax(np.bincount(labeled.flat)[1:]) + 1
    return (labeled == largest).astype(mask.dtype)


def morphological_cleaning(mask: np.ndarray) -> np.ndarray:
    """Apply small 3×3×3 morphological opening and closing"""
    structure = generate_binary_structure(3, 1)
    mask = binary_opening(mask, structure=structure)
    mask = binary_closing(mask, structure=structure)
    return mask.astype(np.uint8)


def postprocess_volume(volume: np.ndarray, num_classes: int, mode: str) -> np.ndarray:
    """Apply postprocessing per class according to selected mode"""
    processed = np.zeros_like(volume, dtype=np.uint8)

    for cls in range(1, num_classes):  # skip background
        mask = (volume == cls)
        if not np.any(mask):
            continue

        if mode == "lcc":
            mask = largest_connected_component(mask)
        elif mode == "morph":
            mask = morphological_cleaning(mask)
        elif mode == "lcc+morph":
            mask = largest_connected_component(mask)
            mask = morphological_cleaning(mask)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from ['lcc', 'morph', 'lcc+morph'].")

        processed[mask > 0] = cls

    return processed


def main():
    parser = argparse.ArgumentParser(
        description="3D post-processing (LCC, Morph, or Both)."
    )
    parser.add_argument("--in_folder", required=True, help="Folder with input .nii.gz volumes")
    parser.add_argument("--out_folder", required=True, help="Folder to save post-processed volumes")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes (default=5)")
    parser.add_argument(
        "--mode",
        type=str,
        default="lcc+morph",
        choices=["lcc", "morph", "lcc+morph"],
        help="Postprocessing mode: lcc | morph | lcc+morph (default: lcc+morph)",
    )

    args = parser.parse_args()

    in_dir = Path(args.in_folder)
    out_dir = Path(args.out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    nii_files = sorted(in_dir.glob("*.nii.gz"))
    print(f"Found {len(nii_files)} volumes in {in_dir}")
    print(f"Running mode: {args.mode}")

    for f in nii_files:
        print(f"Processing {f.name}")
        nii = nib.load(str(f))
        volume = nii.get_fdata().astype(np.uint8)

        processed = postprocess_volume(volume, args.num_classes, args.mode)

        post_nii = nib.Nifti1Image(processed, affine=nii.affine, header=nii.header)
        nib.save(post_nii, out_dir / f.name)
        print(f" Saved cleaned volume to: {out_dir / f.name}")

    print(f"\n Done! Post-processed volumes saved in: {out_dir}")


if __name__ == "__main__":
    main()
