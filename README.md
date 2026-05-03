# Intelligent Customer Assistant System

End-to-end customer assistant berbasis RAG untuk assignment Data Science and Machine Learning Bootcamp. Project ini memakai dataset `prompt,response` sebagai knowledge base, menghasilkan embedding, menyimpan ke PostgreSQL + PgVector, lalu menyediakan chatbot Streamlit yang bisa menjawab dengan konteks retrieval.

## Fitur

- Preprocessing dataset CSV: cleaning, deduplication, validasi panjang teks, kategori, dan tagging.
- Database schema PgVector dengan metadata, category filter, GIN index, dan HNSW vector index.
- Embedding batch menggunakan `sentence-transformers/all-MiniLM-L6-v2`.
- Semantic search dengan cosine distance, threshold relevance, optional category filter, exact-match boost, dan logging.
- RAG pipeline LangChain + Groq API dengan prompt khusus customer service.
- Streamlit chat UI dengan loading indicator, source context, dan fallback search lokal.
- Evaluasi QnA sederhana: hit rate, mean top score, answer similarity, dan latency.

## Setup

1. Buat virtual environment dan install dependency.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Copy environment template.

```powershell
Copy-Item .env.example .env
```

3. Isi `.env` dengan `GROQ_API_KEY` dan `DATABASE_URL`.

4. Siapkan PostgreSQL database dan PgVector.

```sql
CREATE DATABASE customer_assistant;
\c customer_assistant
\i schema.sql
```

## Menyiapkan Knowledge Base

Letakkan dataset original di `data/raw/dataset_assignment.csv`, atau jalankan script dengan path dataset dari Downloads.

```powershell
python scripts\prepare_kb.py --input "C:\Users\bridg\Downloads\dataset_assignment.csv" --output data\processed\knowledge_base.csv --limit 300
```

Script menghasilkan minimal 50 entri knowledge base dengan kolom:

- `source_id`
- `question`
- `answer`
- `content`
- `category`
- `tags`
- `info_type`

Kategori dibuat dari rule sederhana untuk kebutuhan customer assistant:

- `faq`
- `policy`
- `troubleshooting`
- `contact`
- `general`

## Ingestion ke PgVector

```powershell
python scripts\ingest.py --input data\processed\knowledge_base.csv --batch-size 32
```

Kalau PostgreSQL belum tersedia, aplikasi masih bisa memakai file CSV sebagai fallback local search dengan TF-IDF. Untuk penilaian PgVector, jalankan PostgreSQL dan ingestion di atas.

## Menjalankan Chatbot

```powershell
streamlit run app.py
```

Atau gunakan runner Windows yang sudah disiapkan:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_streamlit.ps1 -Port 8501
```

Chatbot akan:

1. Membersihkan pertanyaan user.
2. Melakukan exact match candidate search.
3. Melakukan semantic search di PgVector.
4. Memfilter hasil berdasarkan threshold.
5. Menyusun context untuk prompt customer service.
6. Memanggil Groq LLM untuk menghasilkan jawaban.

Jika `GROQ_API_KEY` belum diisi, sistem mengembalikan jawaban berbasis retrieval agar UI tetap bisa diuji.

## Evaluasi

```powershell
python scripts\evaluate.py --input data\processed\knowledge_base.csv --sample-size 50
```

Metrik:

- `hit_rate_at_k`: seberapa sering dokumen asal ditemukan di top-k.
- `mean_top_score`: rata-rata relevance score top result.
- `mean_answer_similarity`: kemiripan TF-IDF jawaban prediksi dan expected answer.
- `mean_latency_ms`: rata-rata latency retrieval.

## Design Decisions

### Schema

Tabel `knowledge_base` menyimpan teks asli, gabungan `content`, kategori, tag, tipe informasi, metadata JSONB, dan vector embedding. Struktur ini mendukung semantic search sekaligus metadata filtering.

### Indexing

HNSW dipilih karena cocok untuk chatbot interaktif: query latency rendah dan recall stabil tanpa training index. GIN index dipakai untuk `tags` dan `metadata`, sedangkan B-tree index dipakai untuk `category` dan `info_type`.

### Embedding Model

`sentence-transformers/all-MiniLM-L6-v2` dipakai karena ringan, cepat di laptop, dimensi embedding hanya 384, dan cukup baik untuk semantic similarity FAQ/customer support.

### RAG Prompt

Prompt mengarahkan assistant untuk menjawab ramah, ringkas, berbasis konteks, dan mengakui jika konteks tidak cukup. Ini mengurangi halusinasi dan membuat respons lebih cocok untuk customer service.

## Struktur Project

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
