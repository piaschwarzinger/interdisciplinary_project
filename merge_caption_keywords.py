import json
import re
from pathlib import Path
import pandas as pd


def extract_keywords_from_file(folder_path, keyword_file):
    with open(f"{folder_path}/frame_captions.json", "r") as f:
        captions = json.load(f)

    keyword_df = pd.read_csv(keyword_file, encoding = "utf-8-sig").set_index("body_part")
    merged = []

    for entry in captions:
        filename = entry["image"]
        caption = entry["caption"]

        match = re.match(r"frame_\d+_(\w+_\w+|\w+)_([0-9]+)s\.jpg", filename)
        body_part, time = match.groups()
        time = str(int(time))

        keywords = []
        if body_part in keyword_df.index and time in keyword_df.columns:
            raw = keyword_df.loc[body_part, time]
            if pd.notna(raw) and raw.strip():
                keywords = [kw.strip() for kw in raw.split(" and ") if kw.strip()]
                keywords = list(dict.fromkeys(keywords))  # deduplicate

        merged.append({
            "caption": caption,
            "keywords": keywords,
            "body_part": body_part
        })

    with open(f"{folder_path}/merged_caption_keyword.json", "w") as f:
        json.dump(merged, f, indent = 2)

    return merged


# merge body parts under one caption if duplicate
def extract_keywords_from_file_merged_caption(folder_path, keyword_file):
    with open(f"{folder_path}/frame_captions.json", "r") as f:
        captions = json.load(f)

    keyword_df = pd.read_csv(keyword_file, encoding = "utf-8-sig").set_index("body_part")
    grouped = {}

    for entry in captions:
        filename = entry["image"]
        caption = entry["caption"]

        match = re.match(r"frame_\d+_(\w+_\w+|\w+)_([0-9]+)s\.jpg", filename)
        if not match:
            continue
        body_part, time = match.groups()
        time = str(int(time))

        keywords = []
        if body_part in keyword_df.index and time in keyword_df.columns:
            raw = keyword_df.loc[body_part, time]
            if pd.notna(raw) and raw.strip():
                keywords = [kw.strip() for kw in raw.split(" and ") if kw.strip()]
                keywords = list(dict.fromkeys(keywords))  # deduplicate

        if caption not in grouped:
            grouped[caption] = []
        grouped[caption].append({
            "body_part": body_part,
            "keywords": keywords
        })

    # Build merged list with grouped body parts
    merged = []
    for caption, parts in grouped.items():
        merged.append({
            "caption": caption,
            "body_parts": parts  # each entry has {"body_part": ..., "keywords": [...]}
        })

    with open(f"{folder_path}/merged_caption_keyword.json", "w") as f:
        json.dump(merged, f, indent = 2)

    return merged


def build_finetune_dataset(frames_root, target_text_path, keyword_folder, output_path, test = False):
    frames_root = Path(frames_root)
    keyword_folder = Path(keyword_folder)

    # Load video-level descriptions
    with open(target_text_path, "r") as f:
        target_data = json.load(f)
    target_dict = {entry["video_id"]: entry["target_text"] for entry in target_data}

    # Videos to skip for training, but include in test
    skip_videos = {"video_11", "video_12", "video_13"}

    dataset = []

    for keyword_path in sorted(keyword_folder.glob("*.csv")):
        video_id = keyword_path.stem

        # If building train set and this is a test video → skip
        if not test and video_id in skip_videos:
            continue
        # If building test set and this is NOT a test video → skip
        if test and video_id not in skip_videos:
            continue

        folder_path = frames_root / video_id

        merged_data = extract_keywords_from_file_merged_caption(folder_path, keyword_path)
        print(merged_data)
        frame_inputs = []
        seen_body_parts = set()

        for entry in merged_data:
            caption = entry["caption"]
            frame_inputs.append(f"Caption: {caption}.")
            for part in entry["body_parts"]:
                body_part = part["body_part"].replace("_", " ")
                if body_part in seen_body_parts:
                    continue
                seen_body_parts.add(body_part)

                keywords = "; ".join(k.replace("_", " ") for k in part["keywords"])
                frame_inputs.append(
                    f"Body part: {body_part}. Risk factors: {keywords}."
                )

        prompt = ("\n".join(frame_inputs))

        output = target_dict.get(video_id, "No description available.")
        dataset.append({"video_id": video_id, "prompt": prompt, "output": output})

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent = 2)


if __name__ == "__main__":
    data_folder = "data/original_data"
    frame_folder = "high_risk_frames"
    keyword_folder = "keywords"

    build_finetune_dataset(frame_folder, f"{data_folder}/target_text.json", keyword_folder,
                           f"{data_folder}/finetune_train_dataset_original.json")

    build_finetune_dataset(frame_folder, f"{data_folder}/target_text.json", keyword_folder,
                           f"{data_folder}/finetune_test_dataset.json", test = True)
