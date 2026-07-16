# 🎫 Ticket Classification using RAG

An intelligent customer support ticket classification system built using
**Retrieval-Augmented Generation (RAG)**.

Instead of training a dedicated classifier model, this system:
1. Embeds historical, already-labeled tickets into a vector space.
2. Stores them in a vector database (FAISS).
3. When a new ticket arrives, retrieves the most *similar* historical tickets.
4. Uses that retrieved context to decide the category — either via a
   similarity-weighted vote (fully offline) or via an LLM that reasons
   over the retrieved examples (if an OpenAI API key is provided).

This mirrors how RAG is used in production systems for **customer support
automation, IT helpdesk triage, and smart email routing**.

---

## 🧠 How It Works

```
New Ticket
    │
    ▼
 Embedder  (TF-IDF locally, or OpenAI embeddings)
    │
    ▼
FAISS Vector Search  ──►  Top-K most similar historical tickets
    │
    ▼
Classifier
    ├── No API key  → similarity-weighted majority vote (offline)
    └── API key set → LLM reasons over retrieved tickets + picks category
    │
    ▼
Predicted Category + Evidence (which past tickets it matched)
```

The key idea of RAG here: the system doesn't just "guess" a category from
the raw text — it **retrieves real, similar, already-labeled tickets first**
and uses them as grounding context, which makes classification more
explainable (you can see *why* it decided a category) and more adaptable
(add new labeled tickets any time without retraining a model).

---

## 🛠 Tech Stack

| Component        | Choice                                   |
|-------------------|-------------------------------------------|
| Embeddings        | TF-IDF (offline, default) or OpenAI `text-embedding-3-small` |
| Vector Database   | FAISS (`IndexFlatIP` with cosine similarity) |
| LLM (optional)    | OpenAI `gpt-4o-mini` for the final reasoning step |
| Data handling     | Pandas / NumPy |

> **Note on embeddings:** the default embedder is TF-IDF so the whole
> project runs 100% offline with no API costs — great for demos, coursework,
> and grading. Swap in `OpenAIEmbedder` (already implemented in
> `src/embedder.py`) for richer semantic embeddings once you want to go
> further.

---

## 📁 Project Structure

```
ticket-rag-system/
├── data/
│   └── historical_tickets.csv     # labeled historical tickets (50 samples, 5 categories)
├── src/
│   ├── embedder.py                # TF-IDF and OpenAI embedding backends
│   ├── vector_store.py            # FAISS wrapper (add / search / save / load)
│   └── rag_classifier.py          # Retrieval + classification logic
├── main.py                        # CLI: build index / classify / demo
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Setup

```bash
pip install -r requirements.txt
```

(Optional) To use the LLM-based classification instead of the offline
voting fallback, copy `.env.example` to `.env` and add your OpenAI key:

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

---

## ▶️ Usage

**1. Build the vector index from historical tickets:**
```bash
python main.py build --data data/historical_tickets.csv
```

**2. Classify a new ticket:**
```bash
python main.py classify --text "My card was charged twice this month"
```

**3. Run the full demo (build + classify 5 sample tickets):**
```bash
python main.py demo
```

**4. Get JSON output (useful for hooking this into an API/webhook):**
```bash
python main.py classify --text "The app keeps crashing" --json
```

---

## 📊 Categories in the Sample Dataset

- **Billing** — charges, refunds, invoices, payment issues
- **Technical Support** — bugs, crashes, errors, performance issues
- **Account Access** — login, password reset, 2FA, account lockouts
- **Feature Request** — new feature suggestions and enhancements
- **General Inquiry** — pricing, support hours, policies, integrations

You can freely extend `data/historical_tickets.csv` with your own tickets
and categories — just re-run `python main.py build` afterward.

---

## 🔍 Example Output

```
Ticket: My video calls freeze constantly, connection keeps dropping

Predicted Category: Technical Support
Confidence: 0.867
Method: similarity_voting

Retrieved Evidence (top matches):
  [Technical Support] (sim=0.597) Video calls keep disconnecting after a few minutes
  [Account Access]    (sim=0.125) My session keeps logging out automatically every few minutes
  [Technical Support] (sim=0.113) I cannot connect to the API, it keeps returning a timeout error
```

---

## 🎓 Ideas for Extending This (great for a graduation project write-up)

- Swap TF-IDF for `sentence-transformers` or OpenAI embeddings for deeper
  semantic matching (e.g. catching paraphrased tickets that share no keywords).
- Add a Flask/FastAPI endpoint so tickets can be classified via HTTP —
  ideal for "Smart Email Routing" or helpdesk integrations.
- Add an evaluation script that computes accuracy/precision/recall against
  a held-out labeled test set.
- Persist feedback: when a human corrects a wrong prediction, add that
  ticket back into the vector store so the system improves over time
  (a simple form of continual learning).
- Try Pinecone or Chroma instead of FAISS if you want a managed/cloud
  vector database instead of a local index.

---

## 📌 Use Cases

- Customer Support Automation
- IT Helpdesk Classification
- Smart Email Routing
