import json
from transformers import pipeline

folder = "high_risk_frames"
video_folder = f"{folder}/video_11"
fewshot_path = f"{folder}/finetune_dataset.json"

with open(f"{video_folder}/merged_caption_keyword.json", "r") as f:
    data = json.load(f)

with open(fewshot_path, "r") as f:
    fewshot_examples = json.load(f)

generator = pipeline("text2text-generation", model="google/flan-t5-base")

fewshot_text = ""
for i, ex in enumerate(fewshot_examples[:1]):
    fewshot_text += f"Example {i+1}:\n{ex['prompt']}\nExplanation: {ex['output']}\n\n"

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
    "You are an ergonomics expert. Analyze the video observations and provide a risk explanation. "
    "Start with a brief overview, then explain why each body part is at risk based on posture and risk factors.\n\n"
    #+ fewshot_text
    + "Now analyze this video:\n"
    + "\n".join(frame_inputs)
)

print(combined_input)

result = generator(combined_input, max_length=500, do_sample=False)[0]
final_description = result["generated_text"]

with open(f"{video_folder}/video_level_description.json", "w") as f:
    json.dump({"video": "video_11", "description": final_description}, f, indent=2)

print("\n⭐️⭐️ Final Video-Level Ergonomic Description: ⭐️⭐️")
print(final_description)