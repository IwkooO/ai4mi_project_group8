import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import nibabel as nib
from skimage.transform import resize



def recover_labels_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    Custom: Rescale mask labels by dividing by the GCD of all nonzero values.
    This is needed if PNGs were saved with label scaling. Returns the corrected mask.
    """
    if mask.size == 0:
        return mask
    nonzero_vals = np.unique(mask[mask > 0])
    if nonzero_vals.size == 0:
        return mask
    divisor = int(np.gcd.reduce(nonzero_vals.astype(np.int64)))
    divisor = divisor if divisor > 0 else 1
    return np.rint(mask / divisor).astype(np.uint8)




def run(args: argparse.Namespace):
    data_folder: Path = args.data_folder
    dest_folder: Path = args.dest_folder
    num_classes: int = args.num_classes
    grp_regex: str = args.grp_regex
    source_scan_pattern: str = args.source_scan_pattern

    assert data_folder.exists(), f"{data_folder} does not exist"
    dest_folder.mkdir(parents=True, exist_ok=True)

    print(f"Stitching data from {data_folder} -> {dest_folder}")


    # Collect files (only .png for now)
    files = list(data_folder.glob("*.png"))
    pattern = re.compile(grp_regex)
    groups = defaultdict(list)



    def z_index_from_filename(stem: str) -> int:
        """Custom: Parse the Z index from a filename stem (e.g., Patient_01_0023 -> 23)."""
        result = re.search(r"(\d+)$", stem)
        if not result:
            raise Exception(f"Z index not found in filename: {stem}")
        return int(result.group(1))


    for file in files:
        match = pattern.match(file.stem)
        if not match:
            raise Exception(f"Grouping failed for file: {file.name} (pattern: {grp_regex})")
        patient_id = match.group(1)
        z = z_index_from_filename(file.stem)
        groups[patient_id].append((z, file))

    print(f"Found {len(groups)} patient groups")


    for patient_id, slices in groups.items():
        # Sort by Z index
        slices = sorted(slices, key=lambda x: x[0])
        print(f"Patient {patient_id}: {len(slices)} slices to process.")

        # Reference scan
        ref_path = Path(source_scan_pattern.format(id_=patient_id))
        if not ref_path.exists():
            raise Exception(f"Reference scan missing for {patient_id}: {ref_path}")
        ref_img = nib.load(str(ref_path))
        ref_shape = ref_img.shape
        ref_affine = ref_img.affine
        xy_shape = ref_shape[:2]
        expected_slices = ref_shape[2]
        if len(slices) != expected_slices:
            raise Exception(f"Slice count mismatch for {patient_id}: expected {expected_slices}, got {len(slices)}")

        # Allocate output
        mask_vol = np.zeros(ref_shape, dtype=np.uint8)

        for z, file in slices:
            mask = np.array(Image.open(file))
            if mask.ndim == 3:
                if not np.all(mask[..., 0] == mask[..., 1]) or not np.all(mask[..., 0] == mask[..., 2]):
                    raise Exception(f"Non-identical RGB channels in {file.name}, not a valid mask!")
                mask = mask[..., 0]
            mask = mask.astype(np.uint8)
            mask = recover_labels_from_mask(mask)
            if mask.max() >= num_classes:
                raise Exception(f"Label value {mask.max()} in {file.name} exceeds allowed {num_classes-1}")
            if mask.shape != xy_shape:
                mask = resize(mask, output_shape=xy_shape, order=0, mode="edge", preserve_range=True, anti_aliasing=False)
                mask = np.round(mask).astype(np.uint8)
            mask_vol[:, :, z] = mask

        # Final validation
        if mask_vol.shape != ref_shape:
            raise Exception(f"Output shape mismatch for {patient_id}: {mask_vol.shape} vs {ref_shape}")
        if mask_vol.dtype != np.uint8:
            raise Exception(f"Output dtype wrong for {patient_id}: {mask_vol.dtype}")
        vmin, vmax = int(mask_vol.min()), int(mask_vol.max())
        if vmin < 0 or vmax > 255:
            raise Exception(f"Output values out of range for {patient_id}: min={vmin}, max={vmax}")
        if vmax >= num_classes:
            raise Exception(f"Output max label for {patient_id} is {vmax}, exceeds {num_classes-1}")

        # Save
        out_path = dest_folder / f"{patient_id}.nii.gz"
        nib.save(nib.Nifti1Image(mask_vol.astype(np.uint8), affine=ref_affine), str(out_path))
        print(f"Saved mask for {patient_id} at {out_path}")






def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Stich pngs to create 3D masks')
    parser.add_argument('--data_folder',type=Path,required=True,help='name of the data folder with sliced data, eg data/prediction/best_epoch/val')
    parser.add_argument('--dest_folder',type=Path,required=True,help='name of the destination folder with stitched data, eg val/pred')
    parser.add_argument('--num_classes',type=int,required=True,help='number of classes in the segmentation task')
    parser.add_argument('--grp_regex',type=str,default='(Patient_\d\d)_\d\d\d\d',help='regex to group pngs by patient')
    parser.add_argument('--source_scan_pattern',type=str,default='data/train/train/{id_}/GT.nii.gz',help='pattern to the original scans to get original size, eg "data/train/train/{id_}/GT.nii.gz" (with {id_} to be replaced in stitch.py by the PatientID)')


    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    run(get_args())