import os

def check_dataset_pairs(audio_dir, image_dir):
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

    print(f"Total audio files found: {len(audio_files)}")
    print(f"Total image files found: {len(image_files)}")
    print(f"Unmatched audio files (no corresponding image): {len(unmatched_audio)}")
    print(f"Unmatched image files (no corresponding audio): {len(unmatched_image)}")

    if unmatched_audio:
        print("List of unmatched audio files:")
        for f in sorted(unmatched_audio):
            print(f"  {f}")
    if unmatched_image:
        print("List of unmatched image files:")
        for f in sorted(unmatched_image):
            print(f"  {f}")

if __name__ == "__main__":
    # Update these paths to your actual dataset locations
    audio_dir = "data/SpeechToImage/Sound"
    image_dir = "data/SpeechToImage/Photos"

    check_dataset_pairs(audio_dir, image_dir)
