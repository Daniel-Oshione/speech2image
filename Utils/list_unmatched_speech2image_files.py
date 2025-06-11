import os

def list_unmatched_files(audio_dir, image_dir):
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

    print(f"Total audio files: {len(audio_files)}")
    print(f"Total image files: {len(image_files)}")
    print(f"Unmatched audio files ({len(unmatched_audio)}):")
    for f in sorted(unmatched_audio):
        print(f"  {f}")
    print(f"Unmatched image files ({len(unmatched_image)}):")
    for f in sorted(unmatched_image):
        print(f"  {f}")

if __name__ == "__main__":
    audio_dir = "data/SpeechToImage/Sound"  # Update if needed
    image_dir = "data/SpeechToImage/Photos"  # Update if needed
    list_unmatched_files(audio_dir, image_dir)
