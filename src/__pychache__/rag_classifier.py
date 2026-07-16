"""
rag_classifier.py
------------------
The core RAG pipeline:

    new ticket -> embed -> retrieve top-k similar historical tickets
               -> build context from retrieved tickets
               -> classify using either:
                    (a) an LLM (OpenAI) that reasons over the retrieved context, or
                    (b) a similarity-weighted majority vote over retrieved
                        categories (fully offline fallback, no API key needed)

This mirrors real-world RAG classification systems used for support ticket
routing, IT helpdesk triage, and smart email routing.
"""

import os
from dotenv import load_dotenv

load_dotenv()

import json

from collections import defaultdict
from .sentiment import analyze_sentiment
from .emotion import detect_emotion
from src.database import (
    get_feedback_tickets,
    assign_employee
)
from src.explainable_ai import ExplainableAI

from .embedder import get_embedder
from .vector_store import TicketVectorStore
from .ticket_analyzer import TicketAnalyzer


CATEGORY_TO_DEPARTMENT = {
    "Account Access": "Account",
    "Billing": "Finance",
    "Technical Support": "Technical Support",
    "General Inquiry": "Support"
}


class RAGTicketClassifier:
    def __init__(self, embedder_backend: str = "tfidf", top_k: int = 5, use_llm: bool = None):
        self.embedder = get_embedder(embedder_backend)
        self.top_k = top_k
        self.store = None
        self.categories = []
        self.analyzer = TicketAnalyzer()
        self.explainer = ExplainableAI()

        # Auto-detect whether we can use an LLM for the final reasoning step.
        if use_llm is None:
            use_llm = bool(os.getenv("OPENAI_API_KEY"))
        self.use_llm = use_llm

        # Initialize OpenAI client if API key is available
        if self.use_llm:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        print("API Key:", os.getenv("OPENAI_API_KEY"))
        print("use_llm:", self.use_llm)
        print("Has client:", hasattr(self, "client"))

    # ------------------------------------------------------------------ #
    # Index building
    # ------------------------------------------------------------------ #
    def build_index(self, texts, categories, ticket_ids=None):
        """Fit the embedder on historical tickets and build the vector index."""
        if ticket_ids is None:
            ticket_ids = list(range(len(texts)))

        self.categories = sorted(set(categories))

        vectors = self.embedder.fit_transform(texts)

        self.store = TicketVectorStore(dim=vectors.shape[1])

        metadata = [
            {
                "id": tid,
                "text": t,
                "category": c
            }
            for tid, t, c in zip(ticket_ids, texts, categories)
        ]

        self.store.add(vectors, metadata)

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        self.embedder.save(os.path.join(dir_path, "embedder.pkl"))
        self.store.save(
            os.path.join(dir_path, "index.faiss"),
            os.path.join(dir_path, "metadata.pkl"),
        )

    def load(self, dir_path: str, embedder_backend: str = "tfidf"):
        self.embedder = get_embedder(embedder_backend)
        self.embedder.load(os.path.join(dir_path, "embedder.pkl"))
        self.store = TicketVectorStore.load(
            os.path.join(dir_path, "index.faiss"),
            os.path.join(dir_path, "metadata.pkl"),
        )
        self.categories = sorted({m["category"] for m in self.store.metadata})
        return self

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(self, ticket_text: str):
        query_vector = self.embedder.transform([ticket_text])
        return self.store.search(query_vector, top_k=self.top_k)

    # ------------------------------------------------------------------ #
    # Feedback Search
    # ------------------------------------------------------------------ #
    def search_feedback(self, ticket_text):

        feedbacks = get_feedback_tickets()

        if not feedbacks:
            return None

        feedback_texts = [
            f["ticket"]
            for f in feedbacks
        ]

        feedback_vectors = self.embedder.transform(
            feedback_texts
        )

        query_vector = self.embedder.transform(
            [ticket_text]
        )[0]

        similarities = feedback_vectors @ query_vector

        best_index = similarities.argmax()

        best_score = float(similarities[best_index])

        if best_score < 0.85:
            return None

        best_feedback = feedbacks[best_index]

        best_feedback["similarity"] = best_score

        return best_feedback

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #
    def classify(self, ticket_text: str) -> dict:
        retrieved = self.retrieve(ticket_text)

        feedback_match = self.search_feedback(ticket_text)

        used_feedback = False

        if not retrieved:
            return {"category": "Uncategorized", "confidence": 0.0, "evidence": []}

        # Retrieve similar tickets for historical reference
        top_ticket = retrieved[0]

        similarity = float(top_ticket[1])

        print("=" * 50)
        print("Top Similarity:", similarity)
        print("Predicted Category:", top_ticket[0]["category"])
        print("Similar Ticket:", top_ticket[0]["text"])
        print("=" * 50)

        # Threshold for considering a ticket as historically similar
        historical_match = similarity >= 0.70

        if feedback_match:

            used_feedback = True

            result = {
                "category": feedback_match["corrected"],
                "confidence": feedback_match["similarity"],
                "priority": "Medium",
                "reason": "Learned from previous administrator feedback.",
                "solution": [],
                "department": CATEGORY_TO_DEPARTMENT.get(
                    feedback_match["corrected"],
                    "Support"
                ),
                "resolution_time": "2 Hours"
            }

            result["used_feedback"] = True

            result["feedback_similarity"] = round(
                feedback_match["similarity"] * 100,
                2
            )

            result["feedback_ticket"] = feedback_match["ticket"]

            result["feedback_original"] = feedback_match["predicted"]

            result["feedback_corrected"] = feedback_match["corrected"]

        else:

            if self.use_llm:
                result = self._classify_with_llm(
                    ticket_text,
                    retrieved
                )

            else:
                result = self._classify_with_voting(
                    retrieved
                )

            result["used_feedback"] = False

        if "priority" not in result:
            result["priority"] = "Medium"

        if "reason" not in result:
            result["reason"] = ""

        if "solution" not in result:
            result["solution"] = []

        if "department" not in result:
            result["department"] = "Support"

        if "resolution_time" not in result:
            result["resolution_time"] = "Unknown"

        result["summary"] = self.analyzer.summarize(ticket_text)
        result["sentiment"] = self.analyzer.detect_sentiment(ticket_text)
        result["urgency"] = self.analyzer.detect_urgency(ticket_text)

        result["evidence"] = [
            {
                "text": m["text"],
                "category": m["category"],
                "similarity": round(s, 3)
            }
            for m, s in retrieved
        ]

        # Explainable AI
        result["explanation"] = self.explainer.explain(
            ticket_text,
            result["evidence"]
        )

        sent = analyze_sentiment(ticket_text)

        result["sentiment"] = sent["sentiment"]
        result["sentiment_confidence"] = sent["confidence"]

        emotion = detect_emotion(ticket_text)

        result["emotion"] = emotion["emotion"]
        result["emotion_confidence"] = emotion["confidence"]

        result["reply"] = "Thank you for contacting us. Our support team will review your request shortly."

        # Historical ticket reference - defaults (no match found)
        result["historical_found"] = False
        result["historical_ticket"] = None
        result["historical_category"] = None
        result["similarity_score"] = 0.0

        # If historically similar ticket found, update values
        if historical_match:

            result["historical_found"] = True

            result["historical_ticket"] = top_ticket[0]["text"]

            result["historical_category"] = top_ticket[0]["category"]

            result["similarity_score"] = round(similarity * 100, 2)

        result["assigned_to"] = assign_employee(
            result["department"]
        )

        result["confidence"] = max(
            0.0,
            min(float(result["confidence"]), 1.0)
        )

        if "used_feedback" not in result:
            result["used_feedback"] = False

        return result

    def ask_about_ticket(self, ticket, analysis, question):

        if not self.use_llm:

            return (
                "AI Assistant requires an OpenAI API Key "
                "to answer free-form questions."
            )

        prompt = f"""
You are an AI customer support assistant.

A ticket has already been analyzed.

Ticket:
{ticket}

Analysis:

Category: {analysis['category']}
Priority: {analysis['priority']}
Department: {analysis['department']}
Assigned Employee: {analysis['assigned_to']}
Sentiment: {analysis['sentiment']}
Urgency: {analysis['urgency']}
Confidence: {analysis['confidence']}
Reason: {analysis['reason']}
Suggested Solution:
{analysis['solution']}

Customer Question:
{question}

Answer ONLY using the ticket and analysis above.

Do not invent information.

Be short, clear and professional.
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    def _classify_with_voting(self, retrieved) -> dict:
        """
        Offline fallback classifier using similarity voting.
        Also predicts a basic priority and solution.
        """
        print(">>> NEW VOTING FUNCTION <<<")

        print("\nRetrieved Tickets:\n")

        for metadata, similarity in retrieved:
            print("--------------------------------")
            print("Category:", metadata["category"])
            print("Similarity:", similarity)
            print(metadata["text"])

        scores = defaultdict(float)

        for metadata, similarity in retrieved:
            scores[metadata["category"]] += similarity

        best_category = max(scores, key=scores.get)

        total = sum(scores.values()) or 1.0
        confidence = round(scores[best_category] / total, 3)

        # Simple priority estimation
        if confidence >= 0.90:
            priority = "High"
        elif confidence >= 0.75:
            priority = "Medium"
        else:
            priority = "Low"

        best_ticket = retrieved[0][0]

        return {
            "category": best_category,
            "priority": best_ticket.get("priority", "Medium"),
            "department": CATEGORY_TO_DEPARTMENT.get(
                best_category,
                "Support"
            ),
            "resolution_time": best_ticket.get(
                "resolution_time",
                "Unknown"
            ),
            "confidence": confidence,
            "reason": "Predicted using RAG similarity search.",
            "solution": [
                best_ticket.get(
                    "resolution",
                    "Investigate issue."
                )
            ],
            "method": "rag_similarity"
        }

    def _classify_with_llm(self, ticket_text: str, retrieved) -> dict:
        """
        Uses GPT to classify the ticket and return structured JSON.
        """

        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        context_lines = []

        for i, (metadata, similarity) in enumerate(retrieved, start=1):
            context_lines.append(
                f"{i}. Category: {metadata['category']}\n"
                f"Similarity: {similarity:.2f}\n"
                f"Ticket: {metadata['text']}"
            )

        context = "\n\n".join(context_lines)

        prompt = f"""
You are an AI Service Desk Assistant.

Available Categories:
{", ".join(self.categories)}

Retrieved Similar Tickets:

{context}

New Ticket:
{ticket_text}

Analyze the ticket carefully.

Return ONLY valid JSON.

Example:

{{
    "category":"Billing",
    "priority":"High",
    "department":"Finance",
    "resolution_time":"24 Hours",
    "reason":"Duplicate payment.",
    "solution":[
        "Verify payment history",
        "Check duplicate transaction",
        "Issue refund",
        "Notify customer"
    ]
}}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        content = response.choices[0].message.content.strip()

        try:
            data = json.loads(content)

        except Exception:

            vote = self._classify_with_voting(retrieved)

            data = {
                "category": vote["category"],
                "priority": vote["priority"],
                "reason": vote["reason"],
                "solution": vote["solution"]
            }

        if data["category"] not in self.categories:
            data["category"] = self._classify_with_voting(retrieved)["category"]

        confidence = round(retrieved[0][1], 3)

        return {
            "category": data["category"],
            "priority": data["priority"],
            "department": data.get("department", "Support"),
            "resolution_time": data.get("resolution_time", "Unknown"),
            "confidence": confidence,
            "reason": data["reason"],
            "solution": data["solution"],
            "method": "llm_rag"
        }