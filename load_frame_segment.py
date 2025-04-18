import os
from pathlib import Path
import pandas as pd
import numpy as np
from moviepy import VideoFileClip
from PIL import Image


def find_video_path(video_name, video_folder):
    for root, _, files in os.walk(video_folder):
        if video_name in files:
            return os.path.join(root, video_name)


def extract_and_save_frames(csv_path, clip, output_dir):
    df = pd.read_csv(csv_path, encoding = "utf-8-sig")
    os.makedirs(output_dir, exist_ok = True)

    idx = 0
    for _, row in df.iterrows():
        body_part = row["body_part"]
        for t in df.columns:
            if t == "body_part" or pd.isna(row[t]):
                continue
            try:
                time = int(float(t))
                if time >= clip.duration:
                    continue

                frame = clip.get_frame(time)
                fname = f"frame_{idx:02d}_{body_part}_{time}s.jpg"
                fpath = os.path.join(output_dir, fname)
                Image.fromarray(np.uint8(frame)).save(fpath)
                print(f"Saved: {fpath}")
                idx += 1
            except Exception as e:
                print(f"Failed at time {t} for {body_part}: {e}")
                continue


def process_single_video(csv_path, input_folder, video_folder, output_folder):
    video_id = csv_path.replace("video_", "").replace(".csv", "")
    video_name = f"Ergo_Video_{int(video_id)}.mp4"

    video_path = find_video_path(video_name, video_folder)
    if not video_path:
        print(f"Video not found for {csv_path}")
        return

    try:
        clip = VideoFileClip(video_path)
    except Exception as e:
        print(f"Error loading video {video_name}: {e}")
        return

    csv_path = os.path.join(input_folder, csv_path)
    output_dir = os.path.join(output_folder, f"video_{video_id}")
    extract_and_save_frames(csv_path, clip, output_dir)


if __name__ == "__main__":
    input_folder = "keywords"
    video_folder = "video_data"
    output_folder = "high_risk_frames"
    os.makedirs(output_folder, exist_ok = True)

    for csv_path in Path(input_folder).glob("*.csv"):
        process_single_video(csv_path.name, input_folder, video_folder, output_folder)
