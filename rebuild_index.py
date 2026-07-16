import pandas as pd

from src.rag_classifier import RAGTicketClassifier

print("Loading dataset...")

df = pd.read_csv("data/historical_tickets.csv")

print(f"Loaded {len(df)} tickets")

classifier = RAGTicketClassifier(
    embedder_backend="sentence",
    top_k=5
)

classifier.build_index(
    texts=df["text"].tolist(),
    categories=df["category"].tolist()
)

classifier.save("index_store")

print("\n✅ Index rebuilt successfully!")
print(f"Indexed {len(df)} historical tickets.")