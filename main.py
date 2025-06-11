from Utils.helpers import load_flickr8k, load_scenes_classification
from Models.gan_model import build_generator, build_discriminator, GAN
from tensorflow.keras.optimizers import Adam
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
        }
    ]

    speech_audio_data = []
    speech_image_data = []

    print("Loading datasets...")

    # Load and preprocess each dataset separately by type
    for ds in datasets:
        print(f"Loading {ds['name']} dataset...")
        if ds['name'] == 'Flickr8k':
            image_data, captions = load_flickr8k(ds['image_dir'], ds['caption_file'], ds['image_size'])
            # Flickr8k has no audio, so skip audio
            speech_audio_data.append(np.zeros((len(image_data), 20, 44)))  # Dummy audio data with expected shape
            speech_image_data.append(image_data)
        elif ds['name'] == 'SceneClassification':
            image_data, labels = load_scenes_classification(ds['image_dir'], ds['csv_file'], ds['image_size'])
            # SceneClassification has no audio, so skip audio
            speech_audio_data.append(np.zeros((len(image_data), 20, 44)))  # Dummy audio data with expected shape
            speech_image_data.append(image_data)

    # Concatenate speech datasets
    if speech_audio_data and speech_image_data:
        # Filter out empty arrays
        speech_audio_data = [arr for arr in speech_audio_data if arr.size > 0]
        speech_image_data = [arr for arr in speech_image_data if arr.size > 0]
        if speech_audio_data:
            X_audio = np.concatenate(speech_audio_data, axis=0)
        else:
            X_audio = None
        if speech_image_data:
            Y_images = np.concatenate(speech_image_data, axis=0)
        else:
            Y_images = None

        if X_audio is not None and Y_images is not None:
            # Reshape X_audio to add channel dimension if missing
            if len(X_audio.shape) == 3:
                X_audio = X_audio[..., np.newaxis]  # Add channel dimension
            print(f"Total samples loaded: {X_audio.shape[0]}")
            print("Starting GAN training on available datasets...")

            generator = build_generator(input_shape=X_audio.shape[1:])
            discriminator = build_discriminator(input_shape=Y_images.shape[1:])
            gan = GAN(generator, discriminator, lambda_perceptual=0.1)

            g_optimizer = Adam(learning_rate=0.0002)
            d_optimizer = Adam(learning_rate=0.0002)
            gan.compile(g_optimizer, d_optimizer)

            # To avoid batch size mismatch errors, disable validation_split and handle validation manually if needed
            # To avoid batch size mismatch errors, disable validation_split and ensure dataset size is divisible by batch size
            # Trim dataset to multiple of batch size
            batch_size = 64
            num_samples = (X_audio.shape[0] // batch_size) * batch_size
            X_audio_trimmed = X_audio[:num_samples]
            Y_images_trimmed = Y_images[:num_samples]

            import tensorflow as tf
            # Create tf.data.Dataset for custom training loop
            dataset = tf.data.Dataset.from_tensor_slices((X_audio_trimmed, Y_images_trimmed))
            from Models.train import custom_gan_training_loop
            custom_gan_training_loop(gan, dataset, epochs=50, batch_size=batch_size)
            print("GAN training completed.")
        else:
            print("No suitable dataset found for training.")
    else:
        print("No suitable dataset found for training.")

if __name__ == "__main__":
    main()
