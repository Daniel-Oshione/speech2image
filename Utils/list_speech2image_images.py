import os

def list_image_files(image_dir):
    if not os.path.exists(image_dir):
        print(f"Image directory does not exist: {image_dir}")
        return

    files = os.listdir(image_dir)
    if not files:
        print(f"No files found in image directory: {image_dir}")
        return

    print(f"Listing files in {image_dir}:")
    for file in files:
        file_path = os.path.join(image_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"{file} - Size: {size} bytes")
        else:
            print(f"{file} - Not a file (possibly a directory)")

if __name__ == "__main__":
    image_dir = "data/SpeechToImage/Photos"  # Update if needed
    list_image_files(image_dir)
