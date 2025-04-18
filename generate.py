import json
from transformers import pipeline

folder = "high_risk_frames"
video_folder = f"{folder}/video_11"

with open(f"{video_folder}/merged_caption_keyword.json", "r") as f:
    data = json.load(f)

generator = pipeline("text2text-generation", model="google/flan-t5-xl")

frame_inputs = []
seen_body_parts = set()

for entry in data:
    caption = entry["caption"]
    frame_inputs.append(f"Caption: {caption}")

    for part in entry["body_parts"]:
        body_part = part["body_part"].replace("_", " ")
        if body_part in seen_body_parts:
            continue
        seen_body_parts.add(body_part)

        keywords = "; ".join(k.replace("_", " ") for k in part["keywords"])
        frame_inputs.append(
            f"Body part: {body_part}. Risk factors: {keywords}."
        )


combined_input = (
    "You are an expert in ergonomics. Based on the following observations from a workplace video, "
    "generate a detailed ergonomic risk explanation. "
    "Start by briefly describing the general scenario observed in the video. "
    "Then, provide a detailed explanation for each body part, including why it is considered at risk based on the "
    "respective risk factors and posture description.\n\n"
    + "\n".join(frame_inputs)
)

print(combined_input)

# Generate final explanation
result = generator(combined_input, max_length=500, do_sample=False)[0]
final_description = result["generated_text"]

print("\n⭐️⭐️ Final Video-Level Ergonomic Description: ⭐️⭐️")
print(final_description)

# Optionally save to JSON
with open(f"{video_folder}/video_level_description.json", "w") as f:
    json.dump({"video": "video_01", "description": final_description}, f, indent=2)