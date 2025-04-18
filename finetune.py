from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import torch
# --- Config ---
model_name = "google/flan-t5-base"
data_path = "high_risk_frames/finetune_dataset_og_adapted.json"
output_dir = "./ergonomics-t5-finetuned"

torch_device = torch.device("cpu")

# --- Load Dataset ---
dataset = load_dataset("json", data_files=data_path, split="train")

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    model_inputs = tokenizer(
        batch["prompt"], padding="max_length", truncation=True, max_length=512
    )
    labels = tokenizer(
        batch["output"], padding="max_length", truncation=True, max_length=256
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenize, batched=True)

# --- Load Model and Apply LoRA ---
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config)
model = model.to(torch_device)

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_strategy="epoch",
    logging_steps=10,
    report_to="none",
)

model = model.to(torch_device)
# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# --- Train ---
trainer.train()

# --- Save Model ---
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)