from dotenv import load_dotenv
import string
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.controllers.main import search_annoy_index

load_dotenv()

class ProcessText:
    def text_processing(self, text: str):
        # text processing
        return (
            text.lower()
            .replace("\n", " ")
            .translate(str.maketrans("", "", string.punctuation))
        )

    def get_embedding(self, query: str):
        # get an embedding for the given query
        return search_annoy_index(query)
    
    def fine_tuning(self, text_input : str):
        model_path = "./fine_tuned_model"
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # tokenize the input text
        tokenized_input = tokenizer(text_input.text, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to("cuda")

        output = model(**tokenized_input)
        
        prediction = output.logits.argmax(dim=1).item()

        return {"prediction": prediction}

process = ProcessText()