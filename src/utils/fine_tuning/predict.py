from transformers import DistilBertTokenizerFast, AutoModelForSequenceClassification
import torch

def fine_tuning(text_input: str, device: str = "cuda"):
    model_path = "./fine_tuned_model"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Move model to the specified device
    model = model.to(device)

    # Tokenize input and move tensors to the specified device
    tokenized_input = tokenizer(
        text_input,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    # Make prediction
    output = model(**tokenized_input)
    probabilities = torch.sigmoid(output.logits)
    threshold = 0.5 
    predicted_labels = (probabilities > threshold).int()

    # Mapping predicted labels to corresponding names
    label_names = [
        "toxic", "profane", "insult", "hate", "threat", 
        "fraud", "offensive", "selfharm", "harassment"
    ]
    predicted_label_names = [label_names[i] for i in range(len(label_names)) if predicted_labels[0, i] == 1]

    return {"predicted_labels": predicted_label_names}