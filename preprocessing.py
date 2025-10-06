import os
import shutil
from pathlib import Path
import argparse


def copy_subset(split: str, input_folder: Path, output_folder: Path, preprocessed_subfolder: str):
    """
    Copy GT and selected preprocessed images from SEGTHOR_CLEAN to SEGTHOR_PREPROCESSED.
    """
    source_split = input_folder / split
    target_split = output_folder / split

    # Copy GT
    gt_src = source_split / "gt"
    gt_dst = target_split / "gt"
    shutil.copytree(gt_src, gt_dst)

    # Copy selected preprocessed data as 'img'
    preproc_src = source_split / preprocessed_subfolder
    preproc_dst = target_split / "img"
    shutil.copytree(preproc_src, preproc_dst)

    print(f"Copied {split}: GT from {gt_src} and IMG from {preproc_src} to {target_split}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segthor_clean", type=str, required=True, help="Path to SEGTHOR_CLEAN directory")
    parser.add_argument("--output_dir", type=str, default="data/SEGTHOR_PREPROCESSED", help="Destination directory")
    parser.add_argument("--preprocessed_subfolder", type=str, required=True,
                        help="Name of the subfolder inside train/val to use as img (e.g., 'preprocessed3D_window_gamma')")

    args = parser.parse_args()

    input_folder = Path(args.segthor_clean)
    output_folder = Path(args.output_dir)

    for split in ["train", "val"]:
        copy_subset(split, input_folder, output_folder, args.preprocessed_subfolder)

    print(f"Done! Preprocessed dataset available at: {output_folder}")


if __name__ == "__main__":
    main()
