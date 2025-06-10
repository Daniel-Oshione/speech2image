import os
import cv2
import pandas as pd
import numpy as np
import librosa

def load_flickr8k(image_dir, caption_file, image_size):
    # Load captions
    captions = []
    with open(caption_file, 'r') as file:
        for line in file.readlines()[1:]:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                # Skip malformed lines
                continue
            caption = parts[1]
            captions.append(caption)

    # Load images
    image_files = os.listdir(image_dir)
    images = []
    for file in image_files:
        image = cv2.imread(os.path.join(image_dir, file))
        image = cv2.resize(image, image_size)
        images.append(image)

    return np.array(images), captions

def load_scenes_classification(image_dir, csv_file, image_size):
    # Load labels
    df = pd.read_csv(csv_file)
    # Adjust label column name based on your CSV file
    if 'CLASS1' in df.columns:
        labels = df['CLASS1'].tolist()
    elif 'CLASS2' in df.columns:
        labels = df['CLASS2'].tolist()
    else:
        raise ValueError("No suitable label column found in CSV file. Expected 'CLASS1' or 'CLASS2'.")

    # Load images
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    images = []
    for file in image_files:
        image_path = os.path.join(image_dir, file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load image file: {image_path}. Skipping.")
            continue
        image = cv2.resize(image, image_size)
        images.append(image)

    return np.array(images), labels

def load_speech_to_image(audio_dir, image_dir, image_size):
    # Load audio
    audio_data = []
    audio_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            file_path = os.path.join(root, file)
            audio_files.append(os.path.relpath(file_path, audio_dir))
            try:
                # Load audio file using librosa or other audio processing library
                audio, sr = librosa.load(file_path)
                # Preprocess audio data (e.g., extract MFCC features)
                mfcc_features = librosa.feature.mfcc(y=audio, sr=sr)
                audio_data.append(mfcc_features)
            except Exception as e:
                print(f"Warning: Failed to load audio file {file_path}: {e}")
                continue

    # Load images recursively
    image_files = []
    images = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            image_path = os.path.join(root, file)
            image_files.append(os.path.relpath(image_path, image_dir))
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Failed to load image file: {image_path}. Skipping.")
                continue
            image = cv2.resize(image, image_size)
            images.append(image)

    # Check for matching audio and image files by filename (without extension)
    audio_basenames = set([os.path.splitext(os.path.basename(f))[0] for f in audio_files])
    image_basenames = set([os.path.splitext(os.path.basename(f))[0] for f in image_files])
    common_basenames = audio_basenames.intersection(image_basenames)

    print(f"Audio files found: {len(audio_files)}")
    print(f"Image files found: {len(image_files)}")
    print(f"Matching audio-image pairs found: {len(common_basenames)}")

    if not common_basenames:
        print("Warning: No matching audio-image file pairs found in SpeechToImage dataset.")
        return np.array([]), np.array([])

    # Filter audio_data and images to only include matching pairs
    filtered_audio_data = []
    filtered_images = []
    for i, audio_file in enumerate(audio_files):
        base = os.path.splitext(os.path.basename(audio_file))[0]
        if base in common_basenames:
            filtered_audio_data.append(audio_data[i])
    for i, image_file in enumerate(image_files):
        base = os.path.splitext(os.path.basename(image_file))[0]
        if base in common_basenames:
            filtered_images.append(images[i])

    return np.array(filtered_audio_data), np.array(filtered_images)

def load_vidTIMIT(data_dir, image_size):
    # Load audio
    audio_data = []
    audio_count = 0
    for speaker_dir in os.listdir(data_dir):
        speaker_audio_dir = os.path.join(data_dir, speaker_dir, 'audio')
        if not os.path.isdir(speaker_audio_dir):
            print(f"Warning: Audio directory does not exist: {speaker_audio_dir}")
            continue
        audio_files = os.listdir(speaker_audio_dir)
        for file in audio_files:
            file_path = os.path.join(speaker_audio_dir, file)
            try:
                # Load audio file using librosa or other audio processing library
                audio, sr = librosa.load(file_path)
                # Preprocess audio data or extracting MFCC features
                mfcc_features = librosa.feature.mfcc(y=audio, sr=sr)
                audio_data.append(mfcc_features)
                audio_count += 1
            except Exception as e:
                print(f"Warning: Failed to load audio file {file_path}: {e}")
                continue

    # Instead of loading video frames, load only the first frame as a representative image
    video_data = []
    video_count = 0
    for speaker_dir in os.listdir(data_dir):
        speaker_video_dir = os.path.join(data_dir, speaker_dir, 'video')
        if not os.path.isdir(speaker_video_dir):
            print(f"Warning: Video directory does not exist: {speaker_video_dir}")
            continue
        video_files = os.listdir(speaker_video_dir)
        for file in video_files:
            file_path = os.path.join(speaker_video_dir, file)
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                print(f"Warning: Failed to read first frame from video file: {file_path}")
                continue
            # Resize the frame to the desired size
            frame = cv2.resize(frame, image_size)
            video_data.append(frame)
            video_count += 1

    if audio_count == 0 or video_count == 0:
        print(f"Warning: Loaded audio files: {audio_count}, video files: {video_count}. One or both are zero.")
        return np.array([]), np.array([])

    # Pad or truncate audio_data to have consistent shape
    max_len = max(mfcc.shape[1] for mfcc in audio_data)
    padded_audio_data = []
    for mfcc in audio_data:
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        padded_audio_data.append(mfcc)
    audio_data_array = np.array(padded_audio_data)

    if len(video_data) == 0:
        print("Warning: No video data loaded.")
        return audio_data_array, np.array([])

    return audio_data_array, np.array(video_data)
