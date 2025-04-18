import os
import cv2
from pathlib import Path

input_root = Path("video_data")
output_root = Path("video_data/augmented_video_data")
output_root.mkdir(parents=True, exist_ok=True)

video_counter = 14
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

for dirpath, dirnames, filenames in os.walk(input_root):
    # Skip the augmented_video_data folder itself
    dirnames[:] = [d for d in dirnames if (Path(dirpath) / d) != output_root]
    for filename in sorted(filenames):
        if filename.endswith(".mp4"):
            input_path = Path(dirpath) / filename
            cap = cv2.VideoCapture(str(input_path))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            output_filename = f"Ergo_Video_{video_counter}.mp4"
            output_path = output_root / output_filename

            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                flipped_frame = cv2.flip(frame, 1)  # Horizontal flip
                out.write(flipped_frame)

            cap.release()
            out.release()
            print(f"Saved: {output_filename}")
            video_counter += 1
