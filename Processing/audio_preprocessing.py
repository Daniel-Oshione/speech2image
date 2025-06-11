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


def extract_mfcc(audio, sr, n_mfcc=20, n_fft=2048, hop_length=512):
    """
    Extract MFCC features from the audio time series.

    Args:
        audio (np.ndarray): Audio time series.
        sr (int): Sample rate.
        n_mfcc (int): Number of MFCC features to extract.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for FFT.

    Returns:
        mfcc (np.ndarray): MFCC features.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    # Normalize MFCC
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
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
