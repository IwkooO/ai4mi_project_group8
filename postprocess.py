import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure

def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a binary mask"""
    if mask.sum() == 0:
        return mask
    struct = generate_binary_structure(3, 2) 
    labeled, n = label(mask, structure=struct)
    if n == 1:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0                  # ignore background
    keep = sizes.argmax()         # largest non-background label
    return labeled == keep

def postprocess_labels(lbl: np.ndarray, K: int) -> np.ndarray:
    """Apply LCC per class on a 3D label volume."""
    out = np.zeros_like(lbl, dtype=np.uint8)
    for c in range(1, K):         # skip background again
        mask = lbl == c
        mask = largest_component(mask)
        out[mask] = c
    return out

def main():
    ap = argparse.ArgumentParser(description="Post-process predictions: Largest Connected Component per class")
    ap.add_argument("--in_folder",  required=True, help="Input folder with stitched .nii.gz predictions")
    ap.add_argument("--out_folder", required=True, help="Output folder for cleaned predictions")
    ap.add_argument("--num_classes", type=int, default=5, help="Number of classes including background")
    args = ap.parse_args()

    in_dir = Path(args.in_folder)
    out_dir = Path(args.out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in sorted(in_dir.glob("*.nii.gz")):
        print(f"[INFO] Processing {f.name}")
        nii = nib.load(str(f))
        lbl = np.asanyarray(nii.get_fdata()).astype(np.uint8)

        cleaned = postprocess_labels(lbl, K=args.num_classes)

        nib.save(
            nib.Nifti1Image(cleaned, affine=nii.affine, header=nii.header),
            str(out_dir / f.name)
        )

if __name__ == "__main__":
    main()
