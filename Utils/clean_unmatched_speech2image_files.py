import os
import shutil

def move_unmatched_files(audio_dir, image_dir, unmatched_audio, unmatched_image):
    unmatched_audio_dir = os.path.join(audio_dir, 'unmatched_audio')
    unmatched_image_dir = os.path.join(image_dir, 'unmatched_images')

    os.makedirs(unmatched_audio_dir, exist_ok=True)
    os.makedirs(unmatched_image_dir, exist_ok=True)

    # Move unmatched audio files
    for root, _, files in os.walk(audio_dir):
        for file in files:
            base = os.path.splitext(file)[0]
            if base in unmatched_audio:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(unmatched_audio_dir, file)
                print(f"Moving unmatched audio file {src_path} to {dst_path}")
                shutil.move(src_path, dst_path)

    # Move unmatched image files
    for root, _, files in os.walk(image_dir):
        for file in files:
            base = os.path.splitext(file)[0]
            if base in unmatched_image:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(unmatched_image_dir, file)
                print(f"Moving unmatched image file {src_path} to {dst_path}")
                shutil.move(src_path, dst_path)

if __name__ == "__main__":
    audio_dir = "data/SpeechToImage/Sound"  # Update if needed
    image_dir = "data/SpeechToImage/Photos"  # Update if needed

    # Automate loading unmatched lists by running list_unmatched_speech2image_files.py logic here
    audio_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            audio_files.append(os.path.splitext(file)[0])

    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            image_files.append(os.path.splitext(file)[0])

    audio_set = set(audio_files)
    image_set = set(image_files)

    unmatched_audio = audio_set - image_set
    unmatched_image = image_set - audio_set

    print(f"Unmatched audio files ({len(unmatched_audio)}): {sorted(unmatched_audio)}")
    print(f"Unmatched image files ({len(unmatched_image)}): {sorted(unmatched_image)}")

    move_unmatched_files(audio_dir, image_dir, unmatched_audio, unmatched_image)
