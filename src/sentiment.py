from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


def analyze_sentiment(text):

    result = classifier(text)[0]

    label = result["label"]
    confidence = round(result["score"], 3)

    if label == "POSITIVE":
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return {
        "sentiment": sentiment,
        "confidence": confidence
    }