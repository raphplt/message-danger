from annoy import AnnoyIndex
import json

from src.controllers.embedding_text import get_embedding_from_text

def search_annoy_index(
    query_text,
    annoy_db_path="annoy_db.ann",
    embeddings_path="embeddings_data.json",
    n_neighbors=5,
):

    annoy_db = AnnoyIndex(
        768, "euclidean"
    )  
    annoy_db.load(
        annoy_db_path
    ) 

    with open(embeddings_path, "r") as file:
        embeddings_data = json.load(file)

    query_embedding = get_embedding_from_text(query_text)

    neighbor_ids = annoy_db.get_nns_by_vector(
        query_embedding, n=n_neighbors
    )

    print("Voisins trouvés :", neighbor_ids)

query_text = "Fuck bitches I hate them" 
search_annoy_index(query_text)
