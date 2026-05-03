CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_base (
    id BIGSERIAL PRIMARY KEY,
    source_id TEXT UNIQUE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT NOT NULL,
    tags TEXT[] NOT NULL DEFAULT '{}',
    info_type TEXT NOT NULL DEFAULT 'faq',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(384),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_base_category
ON knowledge_base (category);

CREATE INDEX IF NOT EXISTS idx_knowledge_base_info_type
ON knowledge_base (info_type);

CREATE INDEX IF NOT EXISTS idx_knowledge_base_tags
ON knowledge_base USING GIN (tags);

CREATE INDEX IF NOT EXISTS idx_knowledge_base_metadata
ON knowledge_base USING GIN (metadata);

-- HNSW gives strong recall and low-latency retrieval for interactive chat.
-- The vector dimension is 384 for sentence-transformers/all-MiniLM-L6-v2.
CREATE INDEX IF NOT EXISTS idx_knowledge_base_embedding_hnsw
ON knowledge_base
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
