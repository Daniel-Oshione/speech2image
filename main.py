from Utils.helpers import load_flickr8k, load_scenes_classification, load_speech_to_image, load_vidTIMIT
from Models.model import define_model
from Models.train import train_model
import numpy as np

def main():
    # Define dataset directories - update these paths to your actual dataset locations
    datasets = [
        {
            'name': 'Flickr8k',
            'image_dir': 'data/Flickr8k/images',
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
            'image_dir': 'data/SpeechToImage/photos',
            'audio_dir': 'data/SpeechToImage/sound',
            'image_size': (64, 64),
        },
        {
            'name': 'vidTIMIT',
            'data_dir': 'data/vidTIMIT',
            'image_size': (64, 64),
        }
    ]

    all_audio_data = []
    all_image_data = []

    print("Loading datasets...")

    # Load and preprocess each dataset
    for ds in datasets:
        print(f"Loading {ds['name']} dataset...")
        if ds['name'] == 'Flickr8k':
            image_data, captions = load_flickr8k(ds['image_dir'], ds['caption_file'], ds['image_size'])
        elif ds['name'] == 'SceneClassification':
            image_data, labels = load_scenes_classification(ds['image_dir'], ds['csv_file'], ds['image_size'])
        elif ds['name'] == 'SpeechToImage':
            audio_data, image_data = load_speech_to_image(ds['audio_dir'], ds['image_dir'], ds['image_size'])
            all_audio_data.append(audio_data)
            all_image_data.append(image_data)
        elif ds['name'] == 'vidTIMIT':
            audio_data, image_data = load_vidTIMIT(ds['data_dir'], ds['image_size'])
            all_audio_data.append(audio_data)
            all_image_data.append(image_data)

    # Concatenate all datasets if any data loaded
    if not all_audio_data or not all_image_data:
        print("Error: No data loaded from any dataset. Exiting.")
        return

    X_audio = np.concatenate(all_audio_data, axis=0)
    Y_images = np.concatenate(all_image_data, axis=0)

    print(f"Total samples loaded: {X_audio.shape[0]}")

    # Define your model input shape according to the MFCC features shape
    input_shape = X_audio.shape[1:]
    model = define_model(input_shape=input_shape)

    print("Starting model training...")
    train_model(model, X_audio, Y_images, epochs=30, batch_size=64)
    print("Training completed.")

if __name__ == "__main__":
    main()