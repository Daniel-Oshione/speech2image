import librosa
import numpy as np
import os

def load_audio(file_path):
    """
    Load audio file using librosa.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        audio (np.ndarray): Audio time series.
        sr (int): Sample rate.
    """
    try:
        audio, sr = librosa.load(file_path)
    except Exception as err:
        print(f"Error loading audio file {file_path}: {err}")
        raise
    return audio, sr



def extract_mfcc(audio, sr):
    """
    Extract MFCC features from the audio time series.

    Args:
        audio (np.ndarray): Audio time series.
        sr (int): Sample rate.

    Returns:
        mfcc (np.ndarray): MFCC features.
    """
    mfcc = librosa.feature.mfcc(audio, sr=sr)
    return mfcc

def preprocess_audio(file_path):
    """
    Preprocess audio file by extracting MFCC features.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        mfcc (np.ndarray): Preprocessed MFCC features.
    """
    audio, sr = load_audio(file_path)
    mfcc = extract_mfcc(audio, sr)
    return mfcc