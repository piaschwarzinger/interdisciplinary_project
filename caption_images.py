import os
import json
from PIL import Image
from pathlib import Path
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model.to(device), device


def process_single_folder(folder_path, processor, model, device):
    results = []
    output_json = f"{folder_path}/frame_captions.json"

    for image_path in sorted(Path(folder_path).glob("*.jpg")):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images = image, return_tensors = "pt").to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens = 50, num_beams = 5)

        caption = processor.decode(output[0], skip_special_tokens = True)
        print(f"{image_path.name}: {caption}")

        results.append({
            "image": image_path.name,
            "caption": caption
        })

    with open(output_json, "w") as f:
        json.dump(results, f, indent = 2)


if __name__ == "__main__":
    frames_folder = "high_risk_frames"
    processor, model, device = load_blip_model()

    for folder_path in sorted(Path(frames_folder).glob("video_*")):
        print(f"\nCaptions for: {folder_path.name}")
        process_single_folder(folder_path, processor, model, device)
