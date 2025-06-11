import os
import shutil

def revert_unmatched_files(audio_dir, image_dir):
    unmatched_audio_dir = os.path.join(audio_dir, 'unmatched_audio')
    unmatched_image_dir = os.path.join(image_dir, 'unmatched_images')

    # Move unmatched audio files back to original folder
    if os.path.exists(unmatched_audio_dir):
        for file in os.listdir(unmatched_audio_dir):
            src_path = os.path.join(unmatched_audio_dir, file)
            dst_path = os.path.join(audio_dir, file)
            print(f"Reverting unmatched audio file {src_path} to {dst_path}")
            shutil.move(src_path, dst_path)

    # Move unmatched image files back to original folder
    if os.path.exists(unmatched_image_dir):
        for file in os.listdir(unmatched_image_dir):
            src_path = os.path.join(unmatched_image_dir, file)
            dst_path = os.path.join(image_dir, file)
            print(f"Reverting unmatched image file {src_path} to {dst_path}")
            shutil.move(src_path, dst_path)

if __name__ == "__main__":
    audio_dir = "data/SpeechToImage/Sound"  # Update if needed
    image_dir = "data/SpeechToImage/Photos"  # Update if needed
    revert_unmatched_files(audio_dir, image_dir)
