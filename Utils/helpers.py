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
    labels = pd.read_csv(csv_file)['label'].tolist()

    # Load images
    image_files = os.listdir(image_dir)
    images = []
    for file in image_files:
        image = cv2.imread(os.path.join(image_dir, file))
        image = cv2.resize(image, image_size)
        images.append(image)

    return np.array(images), labels

def load_speech_to_image(audio_dir, image_dir, image_size):
    # Load audio
    audio_files = os.listdir(audio_dir)
    audio_data = []
    for file in audio_files:
        # Load audio file using librosa or other audio processing library
        audio, sr = librosa.load(os.path.join(audio_dir, file))
        # Preprocess audio data (e.g., extract MFCC features)
        mfcc_features = librosa.feature.mfcc(audio, sr=sr)
        audio_data.append(mfcc_features)

    # Load images
    image_files = os.listdir(image_dir)
    images = []
    for file in image_files:
        image = cv2.imread(os.path.join(image_dir, file))
        image = cv2.resize(image, image_size)
        images.append(image)

    return np.array(audio_data), np.array(images)

def load_vidTIMIT(data_dir, image_size):
    # Load audio
    audio_data = []
    for speaker_dir in os.listdir(data_dir):
        speaker_audio_dir = os.path.join(data_dir, speaker_dir, 'audio')
        audio_files = os.listdir(speaker_audio_dir)
        for file in audio_files:
            # Load audio file using librosa or other audio processing library
            audio, sr = librosa.load(os.path.join(speaker_audio_dir, file))
            # Preprocess audio data or extracting MFCC features
            mfcc_features = librosa.feature.mfcc(audio, sr=sr)
            audio_data.append(mfcc_features)

# Load video
    video_data = []
    for speaker_dir in os.listdir(data_dir):
        speaker_video_dir = os.path.join(data_dir, speaker_dir, 'video')
        video_files = os.listdir(speaker_video_dir)
        for file in video_files:
            # Load video file using OpenCV
            cap = cv2.VideoCapture(os.path.join(speaker_video_dir, file))
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Resize the frame to the desired size
                frame = cv2.resize(frame, image_size)
                frames.append(frame)
            cap.release()
            # Convert the frames to a numpy array
            video_data.append(np.array(frames))

    return np.array(audio_data), np.array(video_data)