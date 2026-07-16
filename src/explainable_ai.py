from keybert import KeyBERT
from sentence_transformers import SentenceTransformer


class ExplainableAI:

    def __init__(self):
        embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = KeyBERT(model=embedding_model)

    def explain(self, ticket_text, evidence):
        keywords = self.model.extract_keywords(
            ticket_text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=5
        )

        keyword_scores = [
            {"keyword": kw, "score": score}
            for kw, score in keywords
        ]

        reasons = []

        for item in evidence[:3]:
            reasons.append(
                f"Similar to previous '{item['category']}' ticket "
                f"(similarity: {item['similarity']:.1%})"
            )

        categories = [e["category"] for e in evidence]
        dominant = max(set(categories), key=categories.count)

        total_sim = sum(e["similarity"] for e in evidence)
        avg_sim = total_sim / len(evidence) if evidence else 0

        return {
            "keywords": keyword_scores,
            "reasons": reasons,
            "dominant_category": dominant,
            "average_similarity": round(avg_sim * 100, 1),
            "agreement": categories.count(dominant),
            "total_neighbors": len(evidence)
        }