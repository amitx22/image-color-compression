from PIL import Image
import numpy as np


def load_image(uploaded_file):
    """
    Load image and convert it to RGB format.
    """
    image = Image.open(uploaded_file).convert("RGB")
    return image


def image_to_pixels(image):
    """
    Convert PIL image into a 2D array of RGB pixels.
    """
    image_array = np.array(image)

    # Height, Width, RGB
    height, width, channels = image_array.shape

    # Convert into (number_of_pixels, 3)
    pixels = image_array.reshape(-1, 3)

    return pixels, height, width


def pixels_to_image(pixels, height, width):
    """
    Convert RGB pixels back into an image.
    """
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)

    image_array = pixels.reshape(height, width, 3)

    return Image.fromarray(image_array)