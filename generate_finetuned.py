import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# === Configuration ===
model_path = "./ergonomics-t5-finetuned_large"  # Path to your fine-tuned model
base_model_name = "google/flan-t5-large"
model_path = "./ergonomics-t5-finetuned"  # Path to your fine-tuned model
base_model_name = "google/flan-t5-base"

video_folder = Path("high_risk_frames/test/video_11")  # Change as needed

# === Load fine-tuned model ===
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()

# === Use GPU if available ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Load merged ergonomic annotations ===
with open(video_folder / "merged_caption_keyword.json", "r") as f:
    data = json.load(f)

# === Build prompt from frame-level info ===
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

print("\n Prompt:\n")
print(combined_input)

# === Tokenize and generate output ===
inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        do_sample = True,
        temperature = 0.7,
        top_p = 0.85,
        repetition_penalty = 1.2,
        max_length = 500
    )

# === Decode and print result ===
final_description = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n⭐️⭐️ Final Video-Level Ergonomic Description: ⭐️⭐️\n")
print(final_description)

# === Save result ===
output_path = video_folder / "video_level_description.json"
with open(output_path, "w") as f:
    json.dump({"video": video_folder.name, "description": final_description}, f, indent=2)
