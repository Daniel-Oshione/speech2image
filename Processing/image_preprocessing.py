import cv2
import numpy as np
import os


def load_image(file_path):
    """
    Load an image from disk.

    Args:
        file_path (str): Path to the image file.

    Returns:
        image (np.ndarray): Loaded RGB image array.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file does not exist: {file_path}")

    image = cv2.imread(file_path, cv2.IMREAD_COLOR)  # Loads as BGR
    if image is None:
        raise ValueError(f"Failed to load image file: {file_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def resize_image(image, size=(64, 64)):
    """
    Resize image to the specified size.

    Args:
        image (np.ndarray): Image array.
        size (tuple): Desired (width, height).

    Returns:
        np.ndarray: Resized image.
    """
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized_image


def preprocess_image(file_path, size=(64, 64)):
    """
    Load and preprocess an image file.

    Args:
        file_path (str): Path to the image file.
        size (tuple): Output size (width, height).

    Returns:
        np.ndarray: Normalized resized image array (values in [0,1]).
    """
    try:
        image = load_image(file_path)
        resized = resize_image(image, size)
        normalized = resized.astype('float32') / 255.0
    except Exception as err:
        print(f"Error processing image file {file_path}: {err}")
        raise
    return normalized
