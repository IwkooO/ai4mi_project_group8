import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# === CONFIG ===
BASE_DIR = "data/SEGTHOR_CLEAN"
SETS = ["val"]  # or ["train", "val"]
INPUT_SUBFOLDER = "img"
OUTPUT_SUBFOLDER = "preprocessed"
GAMMA = 0.7
LOWER_PERCENTILE = 1
UPPER_PERCENTILE = 99.5
# ==============

def histogram_normalization(image, lower=1, upper=99):
    """
    Normalize image based on intensity percentiles.
    Clips intensities and scales to 0–255 (uint8).
    """
    p1 = np.percentile(image, lower)
    p2 = np.percentile(image, upper)
    if p2 - p1 == 0:
        return np.zeros_like(image, dtype=np.uint8)
    image = np.clip(image, p1, p2)
    image = (image - p1) / (p2 - p1)
    return (image * 255).astype(np.uint8)

def adjust_gamma(image, gamma=0.8):
    """
    Apply gamma correction to enhance mid-tone contrast.
    """
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def preprocess_and_save_images(set_name):
    input_path = os.path.join(BASE_DIR, set_name, INPUT_SUBFOLDER)
    output_path = os.path.join(BASE_DIR, set_name, OUTPUT_SUBFOLDER)
    os.makedirs(output_path, exist_ok=True)

    image_paths = sorted(glob(os.path.join(input_path, "*.png")))

    print(f"[{set_name.upper()}] Found {len(image_paths)} images")

    for img_path in tqdm(image_paths, desc=f"Processing {set_name}", unit="img"):
        filename = os.path.basename(img_path)

        # Read as grayscale float32
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Failed to read: {img_path}")
            continue

        image = image.astype(np.float32)

        # Apply histogram normalization and gamma correction
        image = histogram_normalization(image, LOWER_PERCENTILE, UPPER_PERCENTILE)
        image = adjust_gamma(image, gamma=GAMMA)

        out_path = os.path.join(output_path, filename)
        cv2.imwrite(out_path, image)

    print(f"[{set_name.upper()}] Saved processed images to: {output_path}")

def main():
    for set_name in SETS:
        preprocess_and_save_images(set_name)

if __name__ == "__main__":
    main()
