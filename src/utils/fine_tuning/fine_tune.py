import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Define whether GPU or CPU should be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load datasets
dataset = load_dataset("Intuit-GenSRF/combined_toxicity_profanity_v2_train_eval", split="train[:1%]")
dataset_eval = load_dataset("Intuit-GenSRF/combined_toxicity_profanity_v2_train_eval", split="validation[:1%]")

# Load tokenizer and modele
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
num_labels = len(dataset[0]["encoded_labels"])
print("Number of labels:", num_labels)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)

# Data preprocessing
def preprocess_function(examples):
    labels = examples.pop("encoded_labels")
    # Converting the label list to tensor
    labels = torch.tensor(labels, dtype=torch.float32).to(device)
    tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    tokenized_inputs = {key: value.to(device) for key, value in tokenized_inputs.items()}
    # Verify that the label batch size matches the input batch size
    if labels.size(0) == tokenized_inputs["input_ids"].size(0):
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    return None

# Apply preprocessing function to datasets
train_dataset = dataset.shuffle(seed=42).select([i for i in range(len(dataset)) if i % 8 != 0])
train_dataset = train_dataset.map(preprocess_function, batched=True).filter(lambda x: x is not None)

eval_dataset = dataset_eval.shuffle(seed=42).select([i for i in range(len(dataset_eval)) if i % 8 != 0])
eval_dataset = eval_dataset.map(preprocess_function, batched=True).filter(lambda x: x is not None)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=5e-5,
)

# Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# Save model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")