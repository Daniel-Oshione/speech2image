import os

def list_files_with_paths(directory):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.relpath(os.path.join(root, file), directory))
    return file_list

if __name__ == "__main__":
    audio_dir = "data/SpeechToImage/Sound"
    image_dir = "data/SpeechToImage/Photos"

    audio_files = list_files_with_paths(audio_dir)
    image_files = list_files_with_paths(image_dir)

    print("Audio files:")
    for f in audio_files:
        print(f"  {f}")

    print("\nImage files:")
    for f in image_files:
        print(f"  {f}")
