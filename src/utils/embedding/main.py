import json
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)

def get_embedding(message, num_layers_to_use=4):
    inputs = tokenizer(message, return_tensors="pt")
    outputs = model(**inputs, return_dict=True)

    hidden_states = outputs.hidden_states[-num_layers_to_use:]
    
    combined_embedding = torch.stack(hidden_states).mean(dim=0).mean(dim=1).squeeze().detach().numpy()
    
    return combined_embedding

with open("parsed_dataset.json", "r") as file:
    dataset = json.load(file)

embeddings_data = []
for item in dataset:
    message = item["message"]
    problem_type = item["problem_type"]
    embedding = get_embedding(message)
    embeddings_data.append(
        {
            "id": item["id"],
            "message": message,
            "problem_type": problem_type,
            "embedding": embedding.tolist(),
        }
    )
    print(f"Embedding généré pour le message {item['id']}")

output_filename = "embeddings_data.json"
with open(output_filename, "w") as output_file:
    json.dump(embeddings_data, output_file, indent=2)

print(f"Embeddings sauvegardés dans {output_filename}")
