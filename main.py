"""
main.py
-------
CLI entrypoint for the Ticket Classification RAG system.

Usage:
    # 1. Build the vector index from historical tickets
    python main.py build --data data/historical_tickets.csv

    # 2. Classify a new ticket
    python main.py classify --text "My card was charged twice this month"

    # 3. Run the built-in demo (build + classify a few sample tickets)
    python main.py demo
"""

import argparse
import json
import os
print("######## NEW MAIN ########")

import pandas as pd
from dotenv import load_dotenv

from src.rag_classifier import RAGTicketClassifier

load_dotenv()

INDEX_DIR = "index_store"


def build_index(data_path: str, embedder_backend: str):
    df = pd.read_csv(data_path)
    classifier = RAGTicketClassifier(embedder_backend=embedder_backend)
    classifier.build_index(
        texts=df["text"].tolist(),
        categories=df["category"].tolist(),
        ticket_ids=df["ticket_id"].tolist(),
    )
    classifier.save(INDEX_DIR)
    print(f"Index built from {len(df)} tickets across categories: {classifier.categories}")
    print(f"Saved to ./{INDEX_DIR}/")


def classify_ticket(text: str, embedder_backend: str, top_k: int):
    classifier = RAGTicketClassifier(embedder_backend=embedder_backend, top_k=top_k)
    classifier.load(INDEX_DIR, embedder_backend=embedder_backend)

    result = classifier.classify(text)

    print("\n--- Ticket ---")
    print(text)
    print("\n--- AI Analysis ---")

    print(f"Category   : {result['category']}")

    if result.get("priority"):
        print(f"Priority   : {result['priority']}")

    if result.get("department"):
        print(f"Department : {result['department']}")

    if result.get("resolution_time"):
        print(f"ETA        : {result['resolution_time']}")

    if result.get("confidence") is not None:
        print(f"Confidence : {result['confidence']}")

    if result.get("reason"):
        print(f"Reason     : {result['reason']}")

    if result.get("summary"):
        print(f"Summary    : {result['summary']}")

    if result.get("sentiment"):
        print(f"Sentiment  : {result['sentiment']}")

    if result.get("urgency"):
        print(f"Urgency    : {result['urgency']}")

    if result.get("solution"):
        print("\nSuggested Solution:")
        for step in result["solution"]:
            print(f"  • {step}")

    print(f"\nMethod: {result.get('method')}")

    print("\n--- Retrieved Evidence (top matches) ---")
    for ev in result["evidence"]:
        print(f"  [{ev['category']}] (sim={ev['similarity']}) {ev['text']}")

    return result


def run_demo():
    if not os.path.exists(INDEX_DIR):
        build_index("data/historical_tickets.csv",  "sentence")

    sample_tickets = [
        "I got billed twice this month and need a refund",
        "The app crashes every time I open the settings page",
        "I can't log into my account, password reset isn't working",
        "Please add support for exporting reports to CSV",
        "What are your office hours for customer support",
    ]

    classifier = RAGTicketClassifier(embedder_backend= "sentence", top_k=5)
    classifier.load(INDEX_DIR, embedder_backend= "sentence")

    print("=" * 70)
    print("TICKET CLASSIFICATION RAG SYSTEM - DEMO")
    print("=" * 70)

    for ticket in sample_tickets:
        result = classifier.classify(ticket)
        print("\n" + "=" * 60)
        print(f"Ticket: {ticket}")

        print(f"Category   : {result['category']}")
        print(f"Priority   : {result.get('priority')}")
        print(f"Department : {result.get('department')}")
        print(f"ETA        : {result.get('resolution_time')}")
        print(f"Confidence : {result.get('confidence')}")
        print(f"Method     : {result.get('method')}")

        if result.get("reason"):
            print(f"Reason     : {result['reason']}")

        if result.get("solution"):
            print("\nSuggested Solution:")
            for step in result["solution"]:
                print(f"  • {step}")


def main():
    parser = argparse.ArgumentParser(description="Ticket Classification RAG System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the vector index from historical tickets")
    build_parser.add_argument("--data", default="data/historical_tickets.csv")
    build_parser.add_argument("--embedder", default="sentence", choices=["sentence", "tfidf", "openai"])

    classify_parser = subparsers.add_parser("classify", help="Classify a new ticket")
    classify_parser.add_argument("--text", required=True)
    classify_parser.add_argument("--embedder", default="sentence", choices=["sentence", "tfidf", "openai"])
    classify_parser.add_argument("--top_k", type=int, default=5)
    classify_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    subparsers.add_parser("demo", help="Run a full build + classify demo")

    args = parser.parse_args()

    if args.command == "build":
        build_index(args.data, args.embedder)
    elif args.command == "classify":
        result = classify_ticket(args.text, args.embedder, args.top_k)
        if args.json:
            print("\n" + json.dumps(result, indent=2))
    elif args.command == "demo":
        run_demo()


if __name__ == "__main__":
    main()