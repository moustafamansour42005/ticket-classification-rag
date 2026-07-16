import re


class TicketAnalyzer:

    def summarize(self, text: str) -> str:
        """
        Simple extractive summary.
        Returns the first sentence or first 20 words.
        """

        sentences = re.split(r"[.!?]", text)

        if len(sentences) > 0 and sentences[0].strip():
            return sentences[0].strip()

        words = text.split()

        return " ".join(words[:20])


    def detect_sentiment(self, text: str) -> str:

        text = text.lower()

        angry_words = [
            "angry",
            "terrible",
            "worst",
            "ridiculous",
            "disappointed",
            "frustrated",
            "hate",
            "unacceptable"
        ]

        negative_words = [
            "problem",
            "issue",
            "error",
            "failed",
            "cannot",
            "can't",
            "refund"
        ]

        positive_words = [
            "thanks",
            "great",
            "awesome",
            "perfect",
            "love"
        ]

        if any(word in text for word in angry_words):
            return "Angry"

        if any(word in text for word in negative_words):
            return "Negative"

        if any(word in text for word in positive_words):
            return "Positive"

        return "Neutral"


    def detect_urgency(self, text: str) -> str:

        text = text.lower()

        critical = [
            "production",
            "system down",
            "server down",
            "critical",
            "immediately"
        ]

        high = [
            "urgent",
            "asap",
            "today",
            "blocked"
        ]

        medium = [
            "soon",
            "please"
        ]

        if any(word in text for word in critical):
            return "Critical"

        if any(word in text for word in high):
            return "High"

        if any(word in text for word in medium):
            return "Medium"

        return "Low"