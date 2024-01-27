from dotenv import load_dotenv
from profanity_check import predict_prob
import requests
import os

from src.controllers.fine_tuning.predict import fine_tuning


load_dotenv()


class SaplingsClient:
    def __init__(self):
        self.key = os.getenv("SAPLINGS_API_KEY")

    def tone(self, text: str):
        negatives_tones = ['annoyed', 'disappointed', 'disapproving', 'embarrassed', 'fearful', 'repulsed', 'sad', 'worried', 'angry']

        response = requests.post(
            "https://api.sapling.ai/api/v1/tone",
            json={"key": self.key, "text": text},
        )

        return round(sum(score * (1 if tone in negatives_tones else 0) for score, tone, _ in response.json().get('overall', [])), 3)


    def sentiment(self, text: str):
        response = requests.post(
            "https://api.sapling.ai/api/v1/sentiment",
            json={"key": self.key, "text": text},
        )
        return response.json()

    def check_profanity(self, text: str):
        return round(predict_prob([text])[0], 3)
    
client = SaplingsClient()
