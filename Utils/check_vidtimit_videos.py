import os
import cv2

def check_vidtimit_videos(data_dir):
    print(f"Checking video files in {data_dir} ...")
    for speaker_dir in os.listdir(data_dir):
        speaker_video_dir = os.path.join(data_dir, speaker_dir, 'video')
        if not os.path.isdir(speaker_video_dir):
            print(f"Warning: Video directory does not exist: {speaker_video_dir}")
            continue
        video_files = os.listdir(speaker_video_dir)
        for file in video_files:
            file_path = os.path.join(speaker_video_dir, file)
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"Error: Cannot open video file: {file_path}")
                continue
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                print(f"Warning: Failed to read first frame from video file: {file_path}")
            else:
                print(f"Video file OK: {file_path}")

if __name__ == "__main__":
    data_dir = "data/vidTIMIT"  # Update if needed
    check_vidtimit_videos(data_dir)
