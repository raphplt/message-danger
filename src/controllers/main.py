from annoy import AnnoyIndex
from transformers import BertTokenizer, BertModel
import json
import os

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

import re

def get_embedding_from_text(text):
    normalized_text = re.sub(r"[^\w\s]", "", text.lower())
    
    # tokenize and obtain embeddings
    inputs = tokenizer(normalized_text, return_tensors="pt")
    outputs = model(**inputs)
    
    # Calculate the mean of the last hidden states to get a fixed-size embedding
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embedding

def search_annoy_index(
    query_text,
    annoy_db_path="annoy_db.ann",
    embeddings_path="embeddings_data.json",
    n_neighbors=5,
):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    annoy_db_path = os.path.join(script_dir, annoy_db_path)
    embeddings_path = os.path.join(script_dir, embeddings_path)

    annoy_db = AnnoyIndex(768, "euclidean")
    annoy_db.load(annoy_db_path)

    with open(embeddings_path, "r") as file:
        embeddings_data = json.load(file)

    query_embedding = get_embedding_from_text(query_text)
    print(query_embedding)

    neighbor_ids = annoy_db.get_nns_by_vector(query_embedding, n=n_neighbors)

    neighbors_info = []
    for neighbor_id in neighbor_ids:
        neighbor_data = embeddings_data[neighbor_id]
        neighbor_info = {
            "id": neighbor_data["id"],
            "message": neighbor_data["message"],
            "problem_type": neighbor_data["problem_type"],
        }
        neighbors_info.append(neighbor_info)

    print("Voisins trouvés :", neighbors_info)
    return neighbors_info
