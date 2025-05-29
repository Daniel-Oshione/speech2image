import os

def add_avi_extension(base_dir):
    for speaker in os.listdir(base_dir):
        video_folder = os.path.join(base_dir, speaker, 'video')
        if os.path.isdir(video_folder):
            for filename in os.listdir(video_folder):
                old_path = os.path.join(video_folder, filename)
                print(f"Processing file: {filename}")
                print(f"File path: {old_path}")
                if os.path.isfile(old_path):
                    print(f"Renaming file: {old_path}")
                    new_path = old_path + '.avi'
                    try:
                        os.rename(old_path, new_path)
                        print(f"Renamed {old_path} to {new_path}")
                    except PermissionError:
                        print(f"Permission denied when trying to rename {old_path}.")
                else:
                    print(f"Skipped: {old_path} (not a file)")

if __name__ == "__main__":
    base_dir = 'data/vidTIMIT'  # Update as needed
    add_avi_extension(base_dir)