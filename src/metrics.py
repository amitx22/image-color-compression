import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_mse(original, compressed):
    """
    Calculate Mean Squared Error.
    """

    original = np.array(original).astype(float)
    compressed = np.array(compressed).astype(float)

    mse = np.mean((original - compressed) ** 2)

    return mse


def calculate_psnr(original, compressed):
    """
    Calculate Peak Signal-to-Noise Ratio.
    """

    mse = calculate_mse(original, compressed)

    if mse == 0:
        return float("inf")

    max_pixel = 255.0

    psnr = 10 * np.log10((max_pixel ** 2) / mse)

    return psnr


def calculate_ssim(original, compressed):
    """
    Calculate Structural Similarity Index.
    """

    original = np.array(original)
    compressed = np.array(compressed)

    score = ssim(
        original,
        compressed,
        channel_axis=2,
        data_range=255
    )

    return score