import pandas as pd
from pathlib import Path
import os


def generate_rula_keywords(scores):
    keywords = {}

    # --- Upper arm ---

    main = ""
    if scores['upper_arm'] == 3:
        main = f"moderately above shoulder level (increased strain on shoulder joint)"
    elif scores['upper_arm'] == 4:
        main = f"far above shoulder level (overhead work; high shoulder load)"

    mods = []
    if scores['arm_abducted'] == 1:
        mods.append(f"abducted (Rotator cuff stress; joint instability)")

    if main:
        if mods:
            main += ", with " + " and ".join(mods)
        keywords['upper_arm'] = main
    else:
        keywords['upper_arm'] = " and ".join(mods)

    # --- Lower arm ---
    main = ""
    if scores['lower_arm'] == 2:
        main = f"too extended or overly flexed (fatigue if sustained)"

    keywords['lower_arm'] = main

    # --- Neck ---
    main = ""

    if scores['neck'] == 3:
        main = f"strongly flexed (causing significant muscular load)"
    elif scores['neck'] == 4:
        main = f"neck extension (increasing cervical spine stress)"

    mods = []
    if scores['neck_tilted'] == 1:
        mods.append(f"side-bent (causing muscle imbalance and fatigue)")

    if main:
        if mods:
            main += ", with " + " and ".join(mods)
        keywords['neck'] = main
    else:
        keywords['neck'] = " and ".join(mods)

    # --- Trunk ---
    main = ""

    if scores['trunk'] == 3:
        main = f"bent forward 20 to 60 degrees (causing increased spinal loading)"
    elif scores['trunk'] == 4:
        main = f"severely bent forward (high risk of lower back strain)"

    mods = []
    if scores['trunk_rotated'] == 1:
        mods.append(f"rotated (increasing torsional load on the spine)")
    if scores['trunk_sidebent'] == 1:
        mods.append(f"tilted sideways (leading to muscle imbalance and fatigue)")

    if main:
        if mods:
            main += ", with " + " and ".join(mods)
        keywords['trunk'] = main
    else:
        keywords['trunk'] = " and ".join(mods)

    # --- Lower arm ---
    main = ""
    if scores['legs'] == 2:
        main = f"standing on one leg (asymmetric load on lower body)"

    keywords['legs'] = main

    return keywords


def extract_keywords_over_time(df):
    keyword_dict_per_time = {}

    for col in df.columns:
        time_step_scores = df[col].to_dict()
        keywords = generate_rula_keywords(time_step_scores)
        keyword_dict_per_time[col] = keywords

    result_df = pd.DataFrame.from_dict(keyword_dict_per_time, orient = 'index').T

    # Check if all values are empty or NaN
    if result_df.replace("", pd.NA).isna().all().all():
        for part in result_df.index:
            result_df.loc[part] = "Acceptable conditions, no need for further measures"

    return result_df


def suppress_repeated_keywords(df):
    dedup_df = df.copy()

    for part in dedup_df.index:
        seen = set()
        for col in dedup_df.columns:
            val = dedup_df.loc[part, col]
            if pd.isna(val) or val == "":
                continue
            if val in seen:
                dedup_df.loc[part, col] = ""
            else:
                seen.add(val)

    return dedup_df


def process_single_keyword_csv(csv_path, output_folder):
    df = pd.read_csv(csv_path, sep = ",", encoding = "utf-8-sig", index_col = 0)

    keywords_df = extract_keywords_over_time(df)
    keywords_df = suppress_repeated_keywords(keywords_df)
    keywords_df = keywords_df.reset_index().rename(columns = {'index': 'body_part'})

    output_path = f"{output_folder}/{csv_path.stem}.csv"
    keywords_df.to_csv(output_path, encoding = "utf-8-sig", index = False)


if __name__ == "__main__":
    input_folder = "rula_scores"
    output_folder = "keywords"
    os.makedirs(output_folder, exist_ok = True)

    for csv_path in Path(input_folder).glob("*.csv"):
        if csv_path.name == "rula_rule_counts.csv":
            continue
        process_single_keyword_csv(csv_path, output_folder)