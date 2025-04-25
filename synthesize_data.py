import json
import random
from collections import Counter, defaultdict
import pandas as pd
from generate_keywords import generate_rula_keywords


def define_target_counts():
    return {
        ('upper_arm', 3): 25, ('upper_arm', 4): 29,
        ('trunk', 3): 16, ('trunk', 4): 25,
        ('neck', 3): 30, ('neck', 4): 22,
        ('lower_arm', 2): 24,
        ('legs', 2): 19,
        ('arm_abducted', 1): 29,
        ('neck_tilted', 1): 28,
        ('trunk_rotated', 1): 27,
        ('trunk_sidebent', 1): 30,
    }


def generate_combinations(target_counts, seed = 42):
    random.seed(seed)
    current_counts = defaultdict(int)
    all_pairs = list(target_counts.keys())
    combinations = []

    while any(current_counts[pair] < target_counts[pair] for pair in all_pairs):
        num_parts = random.randint(2, 4)
        available_pairs = [pair for pair in all_pairs if current_counts[pair] < target_counts[pair]]
        sample = available_pairs if len(available_pairs) < num_parts else random.sample(available_pairs, num_parts)
        for pair in sample:
            current_counts[pair] += 1
        combinations.append(sample)

    return combinations


def combinations_to_dataframe(combinations):
    rows = []
    for i, combo in enumerate(combinations, 1):
        video_id = f"video_{i:02}"
        for body_part, score in combo:
            rows.append({"video_id": video_id, "body_part": body_part, "score": score})
    return pd.DataFrame(rows)


def generate_keywords_from_dataframe(df, output_folder):
    keyword_records = []
    for video_id, group in df.groupby("video_id"):
        scores = defaultdict(int)
        for _, row in group.iterrows():
            scores[row["body_part"]] = row["score"]
        keywords = generate_rula_keywords(scores)
        for body_part, explanation in keywords.items():
            if explanation:
                keyword_records.append({
                    "video_id": video_id,
                    "body_part": body_part,
                    "keyword_explanation": explanation
                })

    result_df = pd.DataFrame(keyword_records)

    result_df["video_num"] = result_df["video_id"].str.extract(r'(\d+)').astype(int)
    result_df = result_df.sort_values("video_num").drop(columns = ["video_num"])

    result_df.to_csv(f"{output_folder}/body_parts_keywords.csv", index = False, encoding = "utf-8-sig")
    return result_df


def perform_sanity_check(combinations, target_counts):
    flat_generated = [item for scene in combinations for item in scene]
    count_generated = Counter(flat_generated)

    all_passed = True
    for pair, expected in target_counts.items():
        actual = count_generated.get(pair, 0)
        if actual != expected:
            print(f"Mismatch for {pair}: expected {expected}, got {actual}")
            all_passed = False

    if all_passed:
        print("\nSanity check: All counts match perfectly!")
    else:
        print("\nSanity check: There were mismatches.")


def generate_all_good_keywords(df_keywords, synthetic_folder, num_entries=17):
    all_body_parts = ["upper_arm", "lower_arm", "leg", "neck", "trunk"]

    existing_ids = df_keywords["video_id"].str.extract(r"video_(\d+)").dropna()[0].astype(int)
    start_id = existing_ids.max() + 1

    all_good_rows = []
    for i in range(num_entries):
        video_id = f"video_{start_id + i:03d}"
        for bp in all_body_parts:
            all_good_rows.append({
                "video_id": video_id,
                "body_part": bp,
                "keyword_explanation": "Acceptable conditions, no need for further measures"
            })

    df_all_good = pd.DataFrame(all_good_rows)

    df_combined = pd.concat([df_keywords, df_all_good], ignore_index = True)
    df_combined.to_csv(f"{synthetic_folder}/body_parts_keywords.csv", index = False, encoding = "utf-8-sig")
    return df_combined


def format_body_part(snake_case_name):
    return snake_case_name.replace("_", " ")


def build_prompt(row):
    parts = [
        f"Body part: {format_body_part(bp)} Risk factors: {risk}."
        for bp, risk in zip(row["body_part"], row["keyword_explanation"])
    ]
    return f"Caption: {row['caption']}\n" + "\n".join(parts)


def merge_captions_keywords(captions_df, keywords_df, output_folder):
    captions_df["video_id"] = captions_df["video_id"].apply(lambda x: f"video_{int(x):02}")
    merged_df = pd.merge(keywords_df, captions_df, on = "video_id", how = "left")

    grouped = merged_df.groupby("video_id").agg({
        "caption": "first",
        "body_part": list,
        "keyword_explanation": list
    }).reset_index()

    grouped["prompt"] = grouped.apply(build_prompt, axis = 1)

    json_data = grouped[["video_id", "prompt"]].to_dict(orient = "records")
    json_data_sorted = sorted(json_data, key = lambda x: int(x["video_id"].split("_")[1]))

    output_path = f"{output_folder}/grouped_prompts.json"
    with open(output_path, "w", encoding = "utf-8") as f:
        json.dump(json_data_sorted, f, indent = 2, ensure_ascii = False)

    return json_data_sorted


def merge_train_dataset(prompt_df, target_df, output_folder):
    df_targets = pd.DataFrame(target_df)
    if 'target_text' in df_targets.columns:
        df_targets = df_targets.rename(columns = {"target_text": "output"})
    df_prompts = pd.DataFrame(prompt_df)

    merged_df = pd.merge(df_prompts, df_targets, on = "video_id", how = "inner")
    merged_df["output"] = merged_df["output"].astype(str)

    output_path = f"{output_folder}/finetune_data_synthetic.json"
    merged_df.to_json(output_path, orient = "records", indent = 2)
    return merged_df


def concatenate_finetune_datasets(synthesized_data, original_data, output_path):
    synth_ids = [int(entry["video_id"].split("_")[1]) for entry in synthesized_data]
    max_id = max(synth_ids)

    for i, entry in enumerate(original_data, start = max_id + 1):
        entry["video_id"] = f"video_{i:03d}"

    combined_data = synthesized_data + original_data

    with open(output_path, "w") as f:
        json.dump(combined_data, f, indent = 2)


def generate_synthetic_data():
    original_folder = "data/original_data"
    synthetic_folder = "data/synthetic_data"
    captions_df = pd.read_csv(f"{synthetic_folder}/captions.csv", sep = ";")
    with open(f"{original_folder}/finetune_train_dataset_original.json", "r") as f:
        original_data = json.load(f)

    # Generate required amount of body_part - keyword pairs
    target_counts = define_target_counts()
    combinations = generate_combinations(target_counts)
    df = combinations_to_dataframe(combinations)
    keywords_df = generate_keywords_from_dataframe(df, synthetic_folder)
    perform_sanity_check(combinations, target_counts)
    keywords_df = generate_all_good_keywords(keywords_df, synthetic_folder)

    # Combine body_part - keyword pairs with captions
    prompt_df = merge_captions_keywords(captions_df, keywords_df, synthetic_folder)

    # Manually create target file and merge with prompt data
    with open(f"{synthetic_folder}/generated_target_texts.json", "r") as f:
        target_df = json.load(f)
    synthesized_df = merge_train_dataset(prompt_df, target_df, synthetic_folder)
    synthesized_data = synthesized_df.to_dict(orient = "records")

    # Combine original examples with synthetic examples
    concatenate_finetune_datasets(synthesized_data, original_data, "data/finetune_train_data.json")


if __name__ == "__main__":
    generate_synthetic_data()
