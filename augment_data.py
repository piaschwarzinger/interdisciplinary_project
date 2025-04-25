import os
import cv2
from pathlib import Path


def generate_flipped_videos(input_root, output_subfolder, start_index=14):
    input_root = Path(input_root)
    output_root = input_root / output_subfolder
    output_root.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_files = []
    skip_videos = {"Ergo_Video_11.mp4", "Ergo_Video_12.mp4", "Ergo_Video_13.mp4"}

    # Step 1: Collect sorted video paths (by folder, then file)
    for subfolder in sorted(input_root.iterdir()):
        if subfolder.is_dir() and subfolder.name != output_subfolder:
            for video_file in sorted(subfolder.glob(f"*.mp4")):
                if video_file.name in skip_videos:
                    continue  # Skip this file
                video_files.append(video_file)

    # Step 2: Flip and save in central output folder
    for i, input_path in enumerate(video_files):
        cap = cv2.VideoCapture(str(input_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_filename = f"Ergo_Video_{start_index + i}.mp4"
        output_path = output_root / output_filename
        print(f"{output_filename}")

        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            flipped_frame = cv2.flip(frame, 1)
            out.write(flipped_frame)

        cap.release()
        out.release()


if __name__ == "__main__":
    input_root = "video_data"
    output_subfolder = "augmented_video_data"
    generate_flipped_videos(input_root, output_subfolder)