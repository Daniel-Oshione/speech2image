import os
import shutil

def move_unmatched_files(audio_dir, image_dir, unmatched_audio_files, unmatched_image_files, audio_subfolder="unmatched_audio", image_subfolder="unmatched_images"):
    # Create directories for unmatched files
    audio_unmatched_dir = os.path.join(audio_dir, audio_subfolder)
    image_unmatched_dir = os.path.join(image_dir, image_subfolder)
    os.makedirs(audio_unmatched_dir, exist_ok=True)
    os.makedirs(image_unmatched_dir, exist_ok=True)

    # Move unmatched audio files
    for filename in unmatched_audio_files:
        # Find full path of the unmatched audio file
        for root, _, files in os.walk(audio_dir):
            for file in files:
                if os.path.splitext(file)[0] == filename:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(audio_unmatched_dir, file)
                    print(f"Moving unmatched audio file {src_path} to {dst_path}")
                    shutil.move(src_path, dst_path)
                    break

    # Move unmatched image files
    for filename in unmatched_image_files:
        # Find full path of the unmatched image file
        for root, _, files in os.walk(image_dir):
            for file in files:
                if os.path.splitext(file)[0] == filename:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(image_unmatched_dir, file)
                    print(f"Moving unmatched image file {src_path} to {dst_path}")
                    shutil.move(src_path, dst_path)
                    break

def load_unmatched_lists_from_file(audio_file_path, image_file_path):
    unmatched_audio_files = []
    unmatched_image_files = []

    if os.path.exists(audio_file_path):
        with open(audio_file_path, 'r') as f:
            unmatched_audio_files = [line.strip() for line in f if line.strip()]
    else:
        print(f"Audio unmatched list file not found: {audio_file_path}")

    if os.path.exists(image_file_path):
        with open(image_file_path, 'r') as f:
            unmatched_image_files = [line.strip() for line in f if line.strip()]
    else:
        print(f"Image unmatched list file not found: {image_file_path}")

    return unmatched_audio_files, unmatched_image_files

if __name__ == "__main__":
    # Update these paths to your actual dataset locations
    audio_dir = "data/SpeechToImage/Sound"
    image_dir = "data/SpeechToImage/Photos"

    # Paths to files containing unmatched filenames (one per line)
    audio_unmatched_list_file = "unmatched_audio.txt"
    image_unmatched_list_file = "unmatched_image.txt"

    # Load unmatched filenames from files
    unmatched_audio_files, unmatched_image_files = load_unmatched_lists_from_file(audio_unmatched_list_file, image_unmatched_list_file)

    if not unmatched_audio_files and not unmatched_image_files:
        print("No unmatched files to move. Please create unmatched_audio.txt and unmatched_image.txt with filenames to move.")
    else:
        # Call the function to move unmatched files
        move_unmatched_files(audio_dir, image_dir, unmatched_audio_files, unmatched_image_files)
