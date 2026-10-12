# Intelligent Customer Assistant System

An end-to-end, retrieval-augmented (RAG) customer support chatbot, built for the Data Science and Machine Learning Bootcamp. It turns a `prompt,response` dataset into a knowledge base, embeds it into PostgreSQL + pgvector, and serves a Streamlit chatbot that answers with retrieved context.

## Features

- CSV preprocessing: cleaning, deduplication, text length validation, categorisation and tagging
- pgvector schema with metadata, category filters, a GIN index and an HNSW vector index
- Batch embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Semantic search with cosine distance, a relevance threshold, optional category filtering, exact-match boosting and logging
- RAG pipeline with LangChain and the Groq API, using a customer-service prompt
- Streamlit chat UI with a loading indicator, source context and a local search fallback
- Simple QnA evaluation: hit rate, mean top score, answer similarity and latency

## How a question is answered

1. Clean the user's question.
2. Look for exact-match candidates.
3. Run semantic search in pgvector.
4. Filter results by the relevance threshold.
5. Build the context for the customer-service prompt.
6. Call the Groq LLM to generate the answer.

If `GROQ_API_KEY` isn't set, the system returns a retrieval-based answer so the UI can still be tested.

## Setup

1. Create a virtual environment and install dependencies. The main requirements are enough for the Streamlit demo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For the full pgvector, Groq and local embedding ingestion features:

```powershell
pip install -r requirements-full.txt
```

2. Copy the environment template and fill in `GROQ_API_KEY` and `DATABASE_URL`:

```powershell
Copy-Item .env.example .env
```

3. Prepare PostgreSQL with pgvector:

```sql
CREATE DATABASE customer_assistant;
\c customer_assistant
\i schema.sql
```

## Building the knowledge base

Put the original dataset at `data/raw/dataset_assignment.csv`, or pass its path to the script:

```powershell
python scripts\prepare_kb.py --input data\raw\dataset_assignment.csv --output data\processed\knowledge_base.csv --limit 300
```

The script produces at least 50 knowledge base entries with the columns `source_id`, `question`, `answer`, `content`, `category`, `tags` and `info_type`. Categories come from simple rules suited to a customer assistant: `faq`, `policy`, `troubleshooting`, `contact` and `general`.

## Ingesting into pgvector

```powershell
python scripts\ingest.py --input data\processed\knowledge_base.csv --batch-size 32
```

Without PostgreSQL, the app falls back to TF-IDF search over the CSV file.

## Running the chatbot

```powershell
streamlit run app.py
```

or with the Windows runner:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_streamlit.ps1 -Port 8501
```

## Evaluation

```powershell
python scripts\evaluate.py --input data\processed\knowledge_base.csv --sample-size 50
```

Metrics:

- `hit_rate_at_k`: how often the source document appears in the top-k results
- `mean_top_score`: average relevance score of the top result
- `mean_answer_similarity`: TF-IDF similarity between the predicted and expected answers
- `mean_latency_ms`: average retrieval latency

## Design decisions

**Schema.** The `knowledge_base` table stores the original text, a combined `content` field, category, tags, information type, JSONB metadata and the vector embedding, so it supports semantic search and metadata filtering together.

**Indexing.** HNSW suits an interactive chatbot: low query latency and stable recall without training the index. GIN indexes cover `tags` and `metadata`; B-tree indexes cover `category` and `info_type`.

**Embedding model.** `all-MiniLM-L6-v2` is light, fast on a laptop, has only 384 dimensions and works well for FAQ and support-style semantic similarity.

**RAG prompt.** The prompt asks the assistant to be friendly, concise and grounded in the context, and to say so when the context isn't enough. This reduces hallucination and fits customer service.

## Project structure

```text
app.py
schema.sql
src/
  config.py
  embeddings.py
  local_search.py
  pgvector_store.py
  preprocessing.py
  rag.py
scripts/
  prepare_kb.py
  ingest.py
  evaluate.py
data/
  raw/
  processed/
```
