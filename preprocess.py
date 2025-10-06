import numpy as np
import cv2


def windowing_soft_tissue(volume: np.ndarray, center: float = 60, width: float = 300) -> np.ndarray:
    """
    Applies soft tissue windowing to a CT volume in HU.
    Maps [center - width/2, center + width/2] to [0, 255].
    """
    min_val = center - width / 2
    max_val = center + width / 2
    volume = np.clip(volume, min_val, max_val)
    volume = (volume - min_val) / (max_val - min_val)
    return (volume * 255).astype(np.uint8)


def gamma_correction(image: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """
    Applies gamma correction to a grayscale image.
    gamma < 1 brightens midtones, gamma > 1 darkens midtones.
    """
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def preprocess_ct_volume(volume: np.ndarray, center: float = 60, width: float = 300, gamma: float = 0.8) -> np.ndarray:
    """
    Full preprocessing pipeline: HU windowing + gamma correction.
    Input:
        volume: 3D CT volume in HU
    Output:
        3D volume in uint8 (0-255), ready for slicing or saving
    """
    windowed = windowing_soft_tissue(volume, center=center, width=width)
    corrected = gamma_correction(windowed, gamma=gamma)
    return corrected
