import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# === CONFIGURATION ===
folder = "high_risk_frames"
video_name = "video_11"
video_folder = Path(f"{folder}/test") / video_name

# Path to your fine-tuned model
model_path = "./ergonomics-t5-finetuned"

# === LOAD MODEL & TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
device = 0 if torch.cuda.is_available() else -1

generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

# === LOAD FRAME-LEVEL DATA ===
with open(video_folder / "merged_caption_keyword.json", "r") as f:
    data = json.load(f)

# === BUILD PROMPT ===
frame_inputs = []
seen_body_parts = set()

for entry in data:
    body_part = entry["body_part"].replace("_", " ")
    if body_part in seen_body_parts:
        continue
    seen_body_parts.add(body_part)

    caption = entry["caption"]
    keywords = "; ".join(k.replace("_", " ") for k in entry["keywords"])
    frame_inputs.append(
        f"Caption: {caption}. Body part: {body_part}. Risk factors: {keywords}."
    )

combined_input = (
    "You are an expert in ergonomics. Based on the following observations from a workplace video, "
    "generate a detailed ergonomic risk explanation. "
    "Start by briefly describing the general scenario observed in the video. "
    "Then, provide a detailed explanation for each body part, including why it is considered at risk based on the "
    "respective risk factors and posture description.\n\n"
    + "\n".join(frame_inputs)
)

print("\n📥 Prompt:\n")
print(combined_input)

# === GENERATE EXPLANATION ===
result = generator(
    combined_input,
    max_length=500,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2
)[0]

final_description = result["generated_text"]

# === OUTPUT ===
print("\n⭐️⭐️ Final Video-Level Ergonomic Description: ⭐️⭐️\n")
print(final_description)

# === SAVE TO FILE ===
output_path = video_folder / "video_level_description.json"
with open(output_path, "w") as f:
    json.dump({"video": video_name, "description": final_description}, f, indent=2)

print(f"\n✅ Saved explanation to: {output_path}")