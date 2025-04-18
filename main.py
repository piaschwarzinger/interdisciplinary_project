from pathlib import Path
from generate_keywords import process_single_keyword_csv
from load_frame_segment import process_single_video
from caption_images import load_blip_model, process_single_folder
from merge_caption_keywords import extract_keywords_from_file
import os


def generate_keywords(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok = True)

    for csv_path in Path(input_folder).glob("*.csv"):
        process_single_keyword_csv(csv_path, output_folder)


def load_frame_segment(input_folder, video_folder, output_folder):
    os.makedirs(output_folder, exist_ok = True)

    for csv_path in Path(input_folder).glob("*.csv"):
        process_single_video(csv_path.name, input_folder, video_folder, output_folder)


def caption_images(frames_folder):
    processor, model, device = load_blip_model()

    for folder_path in sorted(Path(frames_folder).glob("video_*")):
        print(f"\nCaptions for: {folder_path.name}")
        process_single_folder(folder_path, processor, model, device)


def merge_caption_keywords(keyword_folder, frames_folder):
    for keyword_path in Path(keyword_folder).glob("*.csv"):
        video_id = keyword_path.stem.replace("video_", "")
        folder_path = f"{frames_folder}/video_{video_id}"
        extract_keywords_from_file(folder_path, keyword_path)


def main():
    scores_folder = "rula_scores"
    keyword_folder = "keywords"
    video_folder = "video_data"
    frames_folder = "high_risk_frames"
    generate_keywords(scores_folder, keyword_folder)
    load_frame_segment(keyword_folder, video_folder, frames_folder)
    caption_images(frames_folder)
    merge_caption_keywords(keyword_folder, frames_folder)


if __name__ == "__main__":
    main()
