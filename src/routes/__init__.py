from fastapi import APIRouter, HTTPException
from src.controllers.fine_tuning.predict import fine_tuning
from src.utils import client
from src.controllers import process

router = APIRouter()


@router.get("/")
def home():
    return {"message": "Hello World"}


@router.get("/tone")
def tone_detector(body: str):
    try:
        updated_text = process.text_processing(body)
        print(updated_text)
        tone_score = client.tone(body)
        return tone_score
    except:
        raise HTTPException(500)


@router.get("/profanity")
def profanity_check(body: str):
    try:
        profanity_percentage = client.check_profanity(body)
        return profanity_percentage
    except:
        raise HTTPException(500)


@router.get("/embedding")
def embedding(body: str):
    try:
        return process.get_embedding(body)
    except:
        raise HTTPException(500)

@router.post("/detect_message")
def detector(body: str):
    try:
        updated_text = process.text_processing(body)
        tone_score = client.tone(updated_text)
        profanity_score = client.check_profanity(updated_text)
        fine_tune_score = fine_tuning(updated_text)

        fine_tune_factor = 0.7 + 0.3 * min(1, len(fine_tune_score) / 8)

        final_score = fine_tune_factor * 100

        final_score += tone_score + profanity_score

        # print("Tone score:", tone_score, "Profanity score:", profanity_score, "Fine tune score:", fine_tune_score)
        # print("Final score:", final_score)

        return {"tone_score": tone_score, "profanity_score": profanity_score, "fine_tune_score": fine_tune_score, "final_score": final_score}
    except:
        raise HTTPException(500)