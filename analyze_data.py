from collections import defaultdict
import pandas as pd
from pathlib import Path

ALL_BODY_PARTS = {
    'upper_arm': [3, 4],
    'arm_abducted': [1],
    'lower_arm': [2],
    'neck': [3, 4],
    'neck_tilted': [1],
    'trunk': [3, 4],
    'trunk_rotated': [1],
    'trunk_sidebent': [1],
    'legs': [2]
}


def check_rula_risks(scores):
    risks = {}

    # --- Upper arm ---
    risks['upper_arm'] = scores.get('upper_arm') in [3, 4]
    risks['arm_abducted'] = scores.get('arm_abducted') == 1

    # --- Lower arm ---
    risks['lower_arm'] = scores.get('lower_arm') == 2

    # --- Neck ---
    risks['neck'] = scores.get('neck') in [3, 4]
    risks['neck_tilted'] = scores.get('neck_tilted') == 1

    # --- Trunk ---
    risks['trunk'] = scores.get('trunk') in [3, 4]
    risks['trunk_rotated'] = scores.get('trunk_rotated') == 1
    risks['trunk_sidebent'] = scores.get('trunk_sidebent') == 1

    # --- Legs ---
    risks['legs'] = scores.get('legs') == 2

    return risks


def count_rule_score_occurrences(csv_path):
    df = pd.read_csv(csv_path, sep=",", encoding="utf-8-sig", index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")

    score_flags = defaultdict(set)

    scores_by_frame = df.transpose().to_dict(orient="index")

    found_any_risk = False

    for score_dict in scores_by_frame.values():
        risks = check_rula_risks(score_dict)

        if any(risks.values()):
            found_any_risk = True

        for body_part, is_risky in risks.items():
            if is_risky:
                score = score_dict.get(body_part)
                if pd.notna(score):
                    score_flags[body_part].add(int(score))

    video_score_counts = defaultdict(lambda: defaultdict(int))

    if not found_any_risk:
        video_score_counts["general"]["info"] = "Acceptable conditions, no need for further measures"
        return video_score_counts

    for body_part, score_set in score_flags.items():
        for score in score_set:
            video_score_counts[body_part][score] += 1

    return video_score_counts


def summarize_rula_scores(input_folder):
    overall_counts = {bp: {s: 0 for s in scores} for bp, scores in ALL_BODY_PARTS.items()}
    acceptable_conditions_count = 0

    for csv_path in sorted(Path(input_folder).glob("video_*.csv")):
        if csv_path.name.endswith(".csv") and "rula_rule_" not in csv_path.name:
            file_counts = count_rule_score_occurrences(csv_path)

            if "general" in file_counts:
                acceptable_conditions_count += 1
                continue  # No risky scores, so skip usual counts

            for body_part, score_map in file_counts.items():
                for score, count in score_map.items():
                    if body_part in overall_counts and score in overall_counts[body_part]:
                        overall_counts[body_part][score] += count

    # Add the special row
    if acceptable_conditions_count > 0:
        overall_counts["acceptable_conditions"] = {1: acceptable_conditions_count}

    # Build DataFrame
    df_final = pd.DataFrame(overall_counts).T.fillna(0).astype(int)
    score_order = sorted(df_final.columns)
    df_final = df_final[score_order]

    df_final.to_csv(f"{input_folder}/rula_rule_counts.csv")
    return df_final


if __name__ == "__main__":
    input_folder = "rula_scores"
    df_summary = summarize_rula_scores(input_folder)
    print(df_summary)