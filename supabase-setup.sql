-- =============================================
-- RAG Small: Supabase pgvector セットアップ
-- Supabase SQL Editor で実行してください
-- =============================================

-- 1. pgvector 拡張を有効化
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. documents テーブル作成
CREATE TABLE IF NOT EXISTS documents (
  id bigint PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  content text NOT NULL,
  metadata jsonb DEFAULT '{}'::jsonb,
  embedding vector(1536),
  created_at timestamptz DEFAULT now()
);

-- 3. ベクトル類似検索用 RPC 関数
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding vector(1536),
  match_threshold float DEFAULT 0.7,
  match_count int DEFAULT 5
)
RETURNS TABLE (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    d.id,
    d.content,
    d.metadata,
    1 - (d.embedding <=> query_embedding) AS similarity
  FROM documents d
  WHERE 1 - (d.embedding <=> query_embedding) > match_threshold
  ORDER BY d.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- 4. HNSW インデックス作成（高速ベクトル検索用）
CREATE INDEX IF NOT EXISTS documents_embedding_idx
  ON documents
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- 5. RLS (Row Level Security) ポリシー
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- サービスロールキーからのフルアクセスを許可
CREATE POLICY "Service role full access"
  ON documents
  FOR ALL
  USING (true)
  WITH CHECK (true);

-- anon キーからの読み取りのみ許可
CREATE POLICY "Anon read access"
  ON documents
  FOR SELECT
  TO anon
  USING (true);
