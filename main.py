from pathlib import Path
from generate_keywords import process_single_keyword_csv
from load_frame_segment import process_single_video
from caption_images import load_blip_model, process_single_folder
from merge_caption_keywords import build_finetune_dataset
from augment_data import generate_flipped_videos
from analyze_data import count_all_keywords
import os

from synthesize_data import generate_synthetic_data


def augment_data(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok = True)

    generate_flipped_videos(input_folder, output_folder)


def generate_keywords(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok = True)

    for csv_path in Path(input_folder).glob("*.csv"):
        if csv_path.name == "rula_rule_counts.csv":
            continue
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


def build_train_test_set(keyword_folder, frame_folder):
    build_finetune_dataset(frame_folder, f"{frame_folder}/target_text.json", keyword_folder,
                               f"{frame_folder}/finetune_train_dataset.json")
    build_finetune_dataset(frame_folder, f"{frame_folder}/target_text.json", keyword_folder,
                               f"{frame_folder}/finetune_test_dataset.json", test = True)


def main():
    scores_folder = "rula_scores"
    keyword_folder = "keywords"
    video_folder = "original_data"
    augmented_video_folder = "augmented_video_data"
    frames_folder = "high_risk_frames"
    # 1. Data augmentation by horizontally flipping the videos (except Video 11, 12, 13 for testing)
    augment_data(video_folder, augmented_video_folder)
    # 2. Compute RULA scores for each body part - use ErgoMaps_Tutorial.ipynb

    # 3. Generate keywords based on the RULA scores
    generate_keywords(scores_folder, keyword_folder)

    # 4. Cut frames from videos for input to Image Captioning model
    load_frame_segment(keyword_folder, video_folder, frames_folder)
    # 5. Generate captions
    caption_images(frames_folder)
    # 6. Manually generate target set and save as "target_text.json"
    # 7. Merge captions and target_text_old.json and build train and test set
    build_train_test_set(keyword_folder, frames_folder)

    # 8.0 Manually generate captions and save as "synthetic_data/captions.csv"
    # 8.1 Manually create target generated_target_texts.json
    # 8.2 Analyze data distribution
    count_all_keywords(scores_folder)
    # 8.3 Generate synthetic train set
    generate_synthetic_data()


if __name__ == "__main__":
    main()
