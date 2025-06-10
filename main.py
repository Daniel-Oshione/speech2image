from Utils.helpers import load_flickr8k, load_scenes_classification, load_speech_to_image, load_vidTIMIT
from Models.model import define_model
from Models.train import train_model
import numpy as np

def main():
    # Define dataset directories - update these paths to your actual dataset locations
    datasets = [
        {
            'name': 'Flickr8k',
            'image_dir': 'data/Flickr8k/Images',
            'caption_file': 'data/Flickr8k/captions.txt',
            'image_size': (64, 64),
        },
        {
            'name': 'SceneClassification',
            'image_dir': 'data/SceneClassification/images',
            'csv_file': 'data/SceneClassification/dataset.csv',
            'image_size': (64, 64),
        },
        {
            'name': 'SpeechToImage',
            'image_dir': 'data/SpeechToImage/Photos',
            'audio_dir': 'data/SpeechToImage/Sound',
            'image_size': (64, 64),
        },
        {
            'name': 'vidTIMIT',
            'data_dir': 'data/vidTIMIT',
            'image_size': (64, 64),
            'use_audio_only': True
        }
    ]

    speech_audio_data = []
    speech_image_data = []

    video_audio_data = []
    video_image_data = []

    print("Loading datasets...")

    # Load and preprocess each dataset separately by type
    for ds in datasets:
        print(f"Loading {ds['name']} dataset...")
        if ds['name'] == 'Flickr8k':
            image_data, captions = load_flickr8k(ds['image_dir'], ds['caption_file'], ds['image_size'])
            # Flickr8k has no audio, so skip audio
        elif ds['name'] == 'SceneClassification':
            image_data, labels = load_scenes_classification(ds['image_dir'], ds['csv_file'], ds['image_size'])
            # SceneClassification has no audio, so skip audio
        elif ds['name'] == 'SpeechToImage':
            audio_data, image_data = load_speech_to_image(ds['audio_dir'], ds['image_dir'], ds['image_size'])
            speech_audio_data.append(audio_data)
            speech_image_data.append(image_data)
        elif ds['name'] == 'vidTIMIT':
            # Use only audio data, skip video data due to corrupted files
            audio_data, _ = load_vidTIMIT(ds['data_dir'], ds['image_size'])
            video_audio_data.append(audio_data)
            # Create dummy images to match audio count
            dummy_images = np.zeros((audio_data.shape[0],) + ds['image_size'] + (3,), dtype=np.float32)
            video_image_data.append(dummy_images)

    # Concatenate speech datasets
    if speech_audio_data and speech_image_data:
        X_audio = np.concatenate(speech_audio_data, axis=0)
        Y_images = np.concatenate(speech_image_data, axis=0)
        # Reshape X_audio to add channel dimension if missing
        if len(X_audio.shape) == 3:
            X_audio = X_audio[..., np.newaxis]  # Add channel dimension
        print(f"Total speech-to-image samples loaded: {X_audio.shape[0]}")
        if X_audio.shape[0] == 0 or Y_images.shape[0] == 0:
            print("Warning: Speech-to-image dataset is empty. Skipping training on this dataset.")
            X_audio = None
            Y_images = None
    else:
        X_audio = None
        Y_images = None

    # Concatenate video datasets
    if video_audio_data and video_image_data:
        X_video_audio = np.concatenate(video_audio_data, axis=0)
        Y_video_images = np.concatenate(video_image_data, axis=0)
        # Reshape X_video_audio to add channel dimension if missing
        if len(X_video_audio.shape) == 3:
            X_video_audio = X_video_audio[..., np.newaxis]  # Add channel dimension
        print(f"Total video samples loaded: {X_video_audio.shape[0]}")
        if X_video_audio.shape[0] == 0 or Y_video_images.shape[0] == 0:
            print("Warning: Video dataset is empty. Skipping training on this dataset.")
            X_video_audio = None
            Y_video_images = None
    else:
        X_video_audio = None
        Y_video_images = None

    # Choose which dataset to train on (speech or video)
    if X_audio is not None and Y_images is not None:
        input_shape = X_audio.shape[1:]
        model = define_model(input_shape=input_shape)
        print("Starting training on speech-to-image dataset...")
        train_model(model, X_audio, Y_images, epochs=30, batch_size=64)
        print("Training completed on speech-to-image dataset.")
    elif X_video_audio is not None and Y_video_images is not None:
        input_shape = X_video_audio.shape[1:]
        model = define_model(input_shape=input_shape)
        print("Starting training on video dataset...")
        train_model(model, X_video_audio, Y_video_images, epochs=30, batch_size=64)
        print("Training completed on video dataset.")
    else:
        print("No suitable dataset found for training.")

if __name__ == "__main__":
    main()
