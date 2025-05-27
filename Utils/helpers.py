import os
import numpy as np
from processing.audio_preprocessing import preprocess_audio
from processing.image_preprocessing import preprocess_image

def load_dataset(audio_dir, image_dir, image_size=(64, 64), audio_extension='.wav', image_extension='.jpg'):
    """
    Load and preprocess paired audio and image files from given directories.

    Args:
        audio_dir (str): Path to the audio files directory.
        image_dir (str): Path to the images directory.
        image_size (tuple): Size to resize images to.
        audio_extension (str): Extension of audio files (default '.wav').
        image_extension (str): Extension of image files (default '.jpg').

    Returns:
        audio_data (np.ndarray): Array of preprocessed audio features.
        image_data (np.ndarray): Array of preprocessed images.
    """
    audio_data = []
    image_data = []

    audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(audio_extension)]

    for audio_file in audio_files:
        base_name = os.path.splitext(audio_file)[0]
        audio_path = os.path.join(audio_dir, audio_file)
        image_file = base_name + image_extension
        image_path = os.path.join(image_dir, image_file)

        if not os.path.exists(image_path):
            print(f"Warning: Image file not found for audio '{audio_file}' - skipping.")
            continue
        
        try:
            mfcc_features = preprocess_audio(audio_path)
            mfcc_features = np.expand_dims(mfcc_features, axis=-1)  # add channel axis

            processed_image = preprocess_image(image_path, size=image_size)
            audio_data.append(mfcc_features)
            image_data.append(processed_image)
        except Exception as e:
            print(f"Error processing pair {audio_file} and {image_file}: {e}")
            continue

    audio_data = np.array(audio_data)
    image_data = np.array(image_data)

    return audio_data, image_data