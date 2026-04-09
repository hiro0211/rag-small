## 2026-04-09

### 作業内容
- Python + Streamlit + LangChain でRAGスモール版を構築
- 当初 Next.js で開始したが、ユーザーの要望で Streamlit に変更
- TDD で全フェーズ���装完了（23テスト、カバレッジ100%）

### 完了フェーズ
- Phase 0: Python プロジェクト初期化（uv, pyproject.toml, pytest）
- Phase 1: Supabase pgvector セットアップ SQL 生成（MCP未接続のためSQL手動実行が必要）
- Phase 2: Embedding パイプライン（load → chunk → embed → store）
- Phase 3: RAG 検索ロジック（ベクトル���索 + プロンプト構築）
- Phase 4: Streamlit チャット UI + ストリーミング応答
- Phase 5: テスト用ドキュメント作成、テスト全通過

### ユーザーが対応すべき残タスク
1. OpenAI API キー取得 → .env.local に記入
2. Supabase プロジェクト作成
3. supabase-setup.sql を Supabase SQL Editor で実行
4. .env.local に Supabase URL/キーを記入
5. `uv run python lib/embedding_pipeline.py --dir ./docs` で Embedding 実行
6. `uv run streamlit run app.py` で動作確認

### 変更ファ���ル一覧
- `pyproject.toml`, `.gitignore`, `.env.local`, `.env.example`
- `lib/supabase_client.py`, `lib/embedding_pipeline.py`, `lib/rag_chain.py`, `lib/chat.py`
- `app.py` (Streamlit UI)
- `tests/test_supabase_client.py`, `tests/test_embedding_pipeline.py`, `tests/test_rag_chain.py`, `tests/test_app.py`
- `supabase-setup.sql`, `docs/test.md`, `USER_TASKS.md`
