from transformers import pipeline

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)


def detect_emotion(text):

    result = emotion_classifier(text)[0][0]

    return {
        "emotion": result["label"],
        "confidence": round(result["score"], 3)
    }